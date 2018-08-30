#pragma once

#include <cassert>
#include <cstdint>
#include <iterator>
#include <type_traits>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <range/v3/span.hpp>

#include "python_util.hpp"

namespace impala
{

class Network
{
public:
	struct Loss
	{
		double v_loss;
		double pi_loss;
		double entropy_loss;
	};

	using State = boost::python::numpy::ndarray;
	using StateTraits = NdArrayTraits<float, 3, 80, 80>;
	using Reward = float;

	Network();
	std::vector<std::tuple<std::int64_t, float>> predict(ranges::span<typename StateTraits::value_type> states);
	Loss train(ranges::span<typename StateTraits::value_type> states, ranges::span<std::int64_t> action_ids, ranges::span<Reward> rewards, ranges::span<float> behaviour_policies, ranges::span<std::int64_t> data_sizes);
	void save(int index);

private:
	boost::python::object m_python_main_ns;
	boost::python::object m_predict_func;
	boost::python::object m_train_func;
	boost::python::object m_save_func;
};

}  // namespace impala
