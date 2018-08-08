#pragma once

#include <cassert>
#include <cstdint>
#include <iterator>
#include <type_traits>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "python_util.hpp"

namespace impala
{

class Network
{
public:
	using State = boost::python::numpy::ndarray;
	using Reward = float;

	Network();
	std::vector<std::tuple<std::int64_t, float>> predict(const State& states);
	std::tuple<double, double, double> train(const State& states, const std::vector<std::int64_t>& action_ids, const std::vector<Reward>& rewards, const std::vector<float>& behaviour_policy, const std::vector<std::int64_t>& data_sizes);
	void save(int index);

private:
	boost::python::object m_python_main_ns;
	boost::python::object m_predict_func;
	boost::python::object m_train_func;
	boost::python::object m_save_func;
};

}  // namespace impala
