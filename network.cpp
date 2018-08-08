#include <experimental/filesystem>

#include "network.hpp"

namespace impala
{

Network::Network()
{
	try {
		namespace fs = std::experimental::filesystem;
		m_python_main_ns = makePythonMainNameSpace();
		boost::python::exec("import sys", m_python_main_ns);
		boost::python::exec(("sys.path.append(\"" + fs::current_path().string() + "\")").data(), m_python_main_ns);
		boost::python::exec_file("train.py", m_python_main_ns);
		m_predict_func = m_python_main_ns["predict_func"];
		m_train_func = m_python_main_ns["train_func"];
		m_save_func = m_python_main_ns["save_model"];
	} catch (boost::python::error_already_set) {
		::PyErr_Print();
		std::terminate();
	}
}

std::vector<std::tuple<std::int64_t, float>> Network::predict(const State& states)
{
	try {
		namespace np = boost::python::numpy;
		const auto batch_size = static_cast<std::size_t>(states.shape(0));
		auto result = m_predict_func(states);
		auto actions = np::from_object(result[0], np::dtype::get_builtin<std::int64_t>(), 1);
		assert(static_cast<std::size_t>(actions.shape(0)) == batch_size);
		assert(actions.strides(0) == sizeof(std::int64_t));
		auto behaviour_policy = np::from_object(result[1], np::dtype::get_builtin<float>(), 1);
		assert(static_cast<std::size_t>(behaviour_policy.shape(0)) == batch_size);
		assert(behaviour_policy.strides(0) == sizeof(float));
		std::vector<std::tuple<std::int64_t, float>> data(batch_size);
		for (std::size_t i = 0; i < batch_size; ++i) {
			auto action_id = reinterpret_cast<std::int64_t*>(actions.get_data())[i];
			auto bp = reinterpret_cast<float*>(behaviour_policy.get_data())[i];
			data.at(i) = std::make_tuple(action_id, bp);
		}
		return data;
	} catch (boost::python::error_already_set) {
		::PyErr_Print();
		std::terminate();
	}
}

std::tuple<double, double, double> Network::train(const State& states, const std::vector<std::int64_t>& action_ids, const std::vector<Reward>& rewards, const std::vector<float>& behaviour_policy, const std::vector<std::int64_t>& data_sizes)
{
	try {
		namespace np = boost::python::numpy;
		const auto t_max = data_sizes.size() - 1;
		const auto batch_size = static_cast<std::size_t>(states.shape(0)) / (t_max + 1);
		assert(action_ids.size() == batch_size * t_max && rewards.size() == batch_size * t_max && behaviour_policy.size() == batch_size * t_max);
		auto action_ids_ndarray = np::from_data(action_ids.data(),
		    np::dtype::get_builtin<std::int64_t>(),
		    boost::python::make_tuple(static_cast<int>(t_max), static_cast<int>(batch_size), 1),
		    boost::python::make_tuple(static_cast<int>(sizeof(std::int64_t) * batch_size), static_cast<int>(sizeof(std::int64_t)), static_cast<int>(sizeof(std::int64_t))),
		    boost::python::object());
		auto rewards_ndarray = np::from_data(rewards.data(),
		    np::dtype::get_builtin<Reward>(),
		    boost::python::make_tuple(static_cast<int>(t_max), static_cast<int>(batch_size), 1),
		    boost::python::make_tuple(static_cast<int>(sizeof(Reward) * batch_size), static_cast<int>(sizeof(Reward)), static_cast<int>(sizeof(Reward))),
		    boost::python::object());
		auto bp_ndarray = np::from_data(behaviour_policy.data(),
		    np::dtype::get_builtin<float>(),
		    boost::python::make_tuple(static_cast<int>(t_max), static_cast<int>(batch_size), 1),
		    boost::python::make_tuple(static_cast<int>(sizeof(float) * batch_size), static_cast<int>(sizeof(float)), static_cast<int>(sizeof(float))),
		    boost::python::object());
		boost::python::list data_sizes_list;
		for (auto&& s : data_sizes) {
			data_sizes_list.append(s);
		}
		auto result = m_train_func(states, action_ids_ndarray, rewards_ndarray, bp_ndarray, data_sizes_list);
		double v_loss = boost::python::extract<double>(result[0]);
		double pi_loss = boost::python::extract<double>(result[1]);
		double entropy_loss = boost::python::extract<double>(result[2]);
		return std::make_tuple(v_loss, pi_loss, entropy_loss);
	} catch (boost::python::error_already_set) {
		::PyErr_Print();
		std::terminate();
	}
}

void Network::save(int index)
{
	try {
		m_save_func(index);
	} catch (boost::python::error_already_set) {
		::PyErr_Print();
		std::terminate();
	}
}

}  // namespace impala
