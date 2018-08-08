#pragma once

#include <cstdlib>
#include <iterator>
#include <type_traits>
#include <vector>

namespace impala
{

template <class T, std::size_t N, std::size_t... Ns>
class TensorRef;

template <class T, std::size_t... Ns>
class TensorRefIterator
{
public:
	using iterator_category = std::input_iterator_tag;
	using value_type = TensorRef<T, Ns...>;
	using difference_type = std::ptrdiff_t;
	using pointer = value_type*;
	using reference = value_type&;

	TensorRefIterator() = default;
	constexpr explicit TensorRefIterator(T* ptr) noexcept : m_ptr{ptr} {}

	constexpr value_type operator*() const noexcept
	{
		return value_type{m_ptr};
	}

private:
	class TensorRefIteratorProxy
	{
	public:
		constexpr explicit TensorRefIteratorProxy(T* ptr) noexcept : m_value{ptr} {}

		constexpr pointer operator->() noexcept
		{
			return &m_value;
		}

	private:
		value_type m_value;
	};

public:
	constexpr TensorRefIteratorProxy operator->() const noexcept
	{
		return TensorRefIteratorProxy{m_ptr};
	}
	constexpr TensorRefIterator& operator++() noexcept
	{
		m_ptr += static_cast<difference_type>((Ns * ...));
		return *this;
	}
	constexpr TensorRefIterator operator++(int) noexcept
	{
		const auto temp = *this;
		++(*this);
		return temp;
	}
	constexpr TensorRefIterator& operator--() noexcept
	{
		m_ptr -= static_cast<difference_type>((Ns * ...));
		return *this;
	}
	constexpr TensorRefIterator operator--(int) noexcept
	{
		const auto temp = *this;
		--(*this);
		return temp;
	}
	constexpr TensorRefIterator& operator+=(difference_type diff) noexcept
	{
		m_ptr += static_cast<difference_type>((Ns * ...)) * diff;
		return *this;
	}
	constexpr TensorRefIterator operator+(difference_type diff) const noexcept
	{
		auto temp = *this;
		temp += diff;
		return temp;
	}
	friend constexpr TensorRefIterator operator+(difference_type diff, const TensorRefIterator& itr) noexcept
	{
		return itr + diff;
	}
	constexpr TensorRefIterator<T, Ns...>& operator-=(difference_type diff) noexcept
	{
		m_ptr -= static_cast<difference_type>((Ns * ...)) * diff;
		return *this;
	}
	constexpr TensorRefIterator<T, Ns...> operator-(difference_type diff) noexcept
	{
		auto temp = *this;
		temp -= diff;
		return temp;
	}
	constexpr difference_type operator-(const TensorRefIterator& other) noexcept
	{
		auto diff = m_ptr - other.m_ptr;
		assert(diff % static_cast<difference_type>((Ns * ...)) == 0);
		return diff / static_cast<difference_type>((Ns * ...));
	}
	constexpr value_type operator[](difference_type n) const noexcept
	{
		return value_type{m_ptr + static_cast<difference_type>((Ns * ...)) * n};
	}
	bool operator==(const TensorRefIterator& other) const noexcept
	{
		return m_ptr == other.m_ptr;
	}
	bool operator!=(const TensorRefIterator& other) const noexcept
	{
		return m_ptr != other.m_ptr;
	}
	bool operator<(const TensorRefIterator& other) const noexcept
	{
		return m_ptr < other.m_ptr;
	}
	bool operator<=(const TensorRefIterator& other) const noexcept
	{
		return m_ptr <= other.m_ptr;
	}
	bool operator>(const TensorRefIterator& other) const noexcept
	{
		return m_ptr > other.m_ptr;
	}
	bool operator>=(const TensorRefIterator& other) const noexcept
	{
		return m_ptr >= other.m_ptr;
	}

private:
	T* m_ptr;
};

template <class T, std::size_t N, std::size_t... Ns>
class TensorRef
{
public:
	using iterator = TensorRefIterator<T, Ns...>;
	using const_iterator = TensorRefIterator<std::add_const_t<T>, Ns...>;

	constexpr explicit TensorRef(T* data) : m_data{data} {}

	TensorRef(const TensorRef&) = delete;
	TensorRef(TensorRef&&) = delete;
	TensorRef& operator=(const TensorRef&) = delete;
	TensorRef& operator=(TensorRef&&) = delete;

	constexpr T* data() noexcept
	{
		return m_data;
	}
	constexpr std::add_const_t<T>* data() const noexcept
	{
		return m_data;
	}
	constexpr std::size_t size() const noexcept
	{
		return N;
	}
	constexpr std::size_t sizeOfAll() const noexcept
	{
		return N * (Ns * ...);
	}
	constexpr TensorRef<T, Ns...> operator[](std::size_t n) noexcept
	{
		assert(n < size());
		return TensorRef<T, Ns...>{data() + n * (Ns * ...)};
	}
	constexpr TensorRef<std::add_const_t<T>, Ns...> operator[](std::size_t n) const noexcept
	{
		assert(n < size());
		return TensorRef<std::add_const_t<T>, Ns...>{data() + n * (Ns * ...)};
	}
	constexpr iterator begin()
	{
		return iterator{data()};
	}
	constexpr iterator end()
	{
		return iterator{data() + N * (Ns * ...)};
	}
	constexpr const_iterator cbegin() const
	{
		return const_iterator{data()};
	}
	constexpr const_iterator cend() const
	{
		return const_iterator{data() + N * (Ns * ...)};
	}
	constexpr const_iterator begin() const
	{
		return cbegin();
	}
	constexpr const_iterator end() const
	{
		return cend();
	}

private:
	T* m_data;
};

template <class T, std::size_t N>
class TensorRef<T, N>
{
public:
	using iterator = T*;
	using const_iterator = std::add_const_t<T>*;

	constexpr explicit TensorRef(T* data) : m_data{data} {}

	TensorRef(const TensorRef&) = delete;
	TensorRef(TensorRef&&) = delete;
	TensorRef& operator=(const TensorRef&) = delete;
	TensorRef& operator=(TensorRef&&) = delete;

	constexpr T* data() noexcept
	{
		return m_data;
	}
	constexpr std::add_const_t<T>* data() const noexcept
	{
		return m_data;
	}
	constexpr std::size_t size() const noexcept
	{
		return N;
	}
	constexpr std::size_t sizeOfAll() const noexcept
	{
		return N;
	}
	constexpr T& operator[](std::size_t n) noexcept
	{
		assert(n < size());
		return data()[n];
	}
	constexpr std::add_const_t<T>& operator[](std::size_t n) const noexcept
	{
		assert(n < size());
		return data()[n];
	}
	constexpr iterator begin()
	{
		return data();
	}
	constexpr iterator end()
	{
		return data() + N;
	}
	constexpr const_iterator cbegin() const
	{
		return data();
	}
	constexpr const_iterator cend() const
	{
		return data() + N;
	}
	constexpr const_iterator begin() const
	{
		return cbegin();
	}
	constexpr const_iterator end() const
	{
		return cend();
	}

private:
	T* m_data;
};

template <class T, std::size_t N, std::size_t... Ns>
class Tensor
{
public:
	static_assert(!std::is_const_v<T>);
	static_assert(((N > 0) && ... && (Ns > 0)));

	using iterator = TensorRefIterator<T, Ns...>;
	using const_iterator = TensorRefIterator<const T, Ns...>;

	Tensor() : m_data(N * (Ns * ...)) {}

private:
	Tensor(const Tensor&) = default;

public:
	static Tensor copy(const Tensor& src)
	{
		return Tensor(src);
	}
	Tensor(Tensor&&) = default;
	Tensor& operator=(const Tensor&) = delete;
	Tensor& operator=(Tensor&&) = default;

	T* data() noexcept
	{
		return m_data.data();
	}
	const T* data() const noexcept
	{
		return m_data.data();
	}
	constexpr std::size_t size() const noexcept
	{
		return N;
	}
	constexpr std::size_t sizeOfAll() const noexcept
	{
		return N * (Ns * ...);
	}
	TensorRef<T, Ns...> operator[](std::size_t n) noexcept
	{
		assert(n < size());
		return TensorRef<T, Ns...>{data() + n * (Ns * ...)};
	}
	TensorRef<const T, Ns...> operator[](std::size_t n) const noexcept
	{
		assert(n < size());
		return TensorRef<const T, Ns...>{data() + n * (Ns * ...)};
	}
	iterator begin()
	{
		return iterator{data()};
	}
	iterator end()
	{
		return iterator{data() + N * (Ns * ...)};
	}
	const_iterator cbegin() const
	{
		return const_iterator{data()};
	}
	const_iterator cend() const
	{
		return const_iterator{data() + N * (Ns * ...)};
	}
	const_iterator begin() const
	{
		return cbegin();
	}
	const_iterator end() const
	{
		return cend();
	}

private:
	std::vector<T> m_data;
};

template <class T, std::size_t N>
class Tensor<T, N>
{
public:
	static_assert(!std::is_const_v<T>);
	static_assert(N > 0);

	using iterator = T*;
	using const_iterator = const T*;

	Tensor() : m_data(N) {}
	T* data() noexcept
	{
		return m_data.data();
	}
	const T* data() const noexcept
	{
		return m_data.data();
	}
	constexpr std::size_t size() const noexcept
	{
		return N;
	}
	constexpr std::size_t sizeOfAll() const noexcept
	{
		return N;
	}
	T& operator[](std::size_t n) noexcept
	{
		assert(n < size());
		return m_data[n];
	}
	const T& operator[](std::size_t n) const noexcept
	{
		assert(n < size());
		return m_data[n];
	}
	iterator begin()
	{
		return data();
	}
	iterator end()
	{
		return data() + N;
	}
	const_iterator cbegin() const
	{
		return data();
	}
	const_iterator cend() const
	{
		return data() + N;
	}
	const_iterator begin() const
	{
		return cbegin();
	}
	const_iterator end() const
	{
		return cend();
	}

private:
	std::vector<T> m_data;
};

}  // namespace impala
