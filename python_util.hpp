#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "tensor.hpp"

namespace impala
{

class PythonInitializer
{
public:
	PythonInitializer(bool init_signal_handler = true)
	{
		assert(!::Py_IsInitialized());
		::Py_InitializeEx(init_signal_handler ? 0 : 1);
		boost::python::numpy::initialize();
	}
	~PythonInitializer()
	{
		::Py_FinalizeEx();
	}
};

inline boost::python::object makePythonMainNameSpace()
{
	return boost::python::import("__main__").attr("__dict__");
}

namespace detail
{

template <std::size_t Current, std::size_t M, std::size_t... Ms, std::size_t... Strides>
inline boost::python::tuple stridesOfNdArrayHelper(std::index_sequence<M, Ms...>, std::index_sequence<Strides...>)
{
	if constexpr (sizeof...(Ms) == 0) {
		return boost::python::make_tuple(static_cast<int>(Strides)..., static_cast<int>(Current / M));
	} else {
		return stridesOfNdArrayHelper<Current / M>(std::index_sequence<Ms...>{}, std::index_sequence<Strides..., Current / M>{});
	}
}

}  // namespace detail

template <class T, std::size_t... Ns>
class NdArrayTraits
{
public:
	static inline constexpr std::size_t size_of_all = (Ns * ...);

	static boost::python::tuple shapeOfNdArray()
	{
		return boost::python::make_tuple(static_cast<int>(Ns)...);
	}
	static boost::python::tuple shapeOfBatchedNdArray(std::size_t batch_size)
	{
		return boost::python::make_tuple(static_cast<int>(batch_size), static_cast<int>(Ns)...);
	}
	static boost::python::tuple stridesOfNdArray()
	{
		return detail::stridesOfNdArrayHelper<sizeof(T) * size_of_all>(std::index_sequence<Ns...>{}, std::index_sequence<>{});
	}
	static boost::python::tuple stridesOfBatchedNdArray()
	{
		return detail::stridesOfNdArrayHelper<sizeof(T) * size_of_all>(std::index_sequence<1, Ns...>{}, std::index_sequence<>{});
	}

	static boost::python::numpy::ndarray convertToNdArray(Tensor<T, Ns...>& tensor)
	{
		namespace np = boost::python::numpy;
		return np::from_data(tensor.data(), np::dtype::get_builtin<T>(), shapeOfNdArray(), stridesOfNdArray(), boost::python::object());
	}

	template <class InputIterator,
	    std::enable_if_t<
	        std::conjunction_v<
	            std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<InputIterator>::iterator_category>,
	            std::is_convertible<typename std::iterator_traits<InputIterator>::reference, const Tensor<T, Ns...>&>>,
	        std::nullptr_t> = nullptr>
	static std::vector<T> makeBufferForBatch(InputIterator first, InputIterator last)
	{
		const auto batch_size = static_cast<std::size_t>(std::distance(first, last));
		std::vector<T> buffer(batch_size * size_of_all);
		auto dest = buffer.begin();
		for (; first != last; ++first) {
			const Tensor<T, Ns...>& src = *first;
			std::copy_n(src.data(), size_of_all, dest);
			dest += size_of_all;
		}
		return buffer;
	}
	template <class InputIterator, class Callback,
	    std::enable_if_t<
	        std::conjunction_v<
	            std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<InputIterator>::iterator_category>,
	            std::is_invocable<Callback, typename std::iterator_traits<InputIterator>::reference, TensorRef<T, Ns...>&>>,
	        std::nullptr_t> = nullptr>
	static std::vector<T> makeBufferForBatch(InputIterator first, InputIterator last, Callback&& callback)
	{
		const auto batch_size = static_cast<std::size_t>(std::distance(first, last));
		std::vector<T> buffer(batch_size * size_of_all);
		auto dest = buffer.data();
		for (; first != last; ++first) {
			TensorRef<T, Ns...> tensor_ref{dest};
			callback(*first, tensor_ref);
			dest += size_of_all;
		}
		return buffer;
	}
	static boost::python::numpy::ndarray convertToBatchedNdArray(std::vector<T>& buffer)
	{
		assert(buffer.size() % size_of_all == 0);
		const auto batch_size = buffer.size() / size_of_all;
		namespace np = boost::python::numpy;
		return np::from_data(buffer.data(), np::dtype::get_builtin<T>(), shapeOfBatchedNdArray(batch_size), stridesOfBatchedNdArray(), boost::python::object());
	}
};


}  // namespace impala
