#include <range/v3/view/indices.hpp>

#include "network.hpp"

namespace impala
{

Network::Network()
{
	try {
		m_python_main_ns = makePythonMainNameSpace();
		boost::python::exec_file("train.py", m_python_main_ns);
		m_predict_func = m_python_main_ns["predict_func"];
		m_train_func = m_python_main_ns["train_func"];
		m_save_func = m_python_main_ns["save_model"];
	} catch (boost::python::error_already_set) {
		::PyErr_Print();
		std::terminate();
	}
}

std::vector<std::tuple<std::int64_t, float>> Network::predict(ranges::span<typename StateTraits::value_type> states)
{
	try {
		namespace np = boost::python::numpy;
		const auto batch_size = static_cast<std::size_t>(states.size()) / StateTraits::size_of_all;
		auto states_ndarray = StateTraits::convertToBatchedNdArray(states);
		auto result = m_predict_func(states_ndarray);
		auto actions = np::from_object(result[0], np::dtype::get_builtin<std::int64_t>(), 1);
		assert(static_cast<std::size_t>(actions.shape(0)) == batch_size);
		assert(actions.strides(0) == sizeof(std::int64_t));
		auto behaviour_policy = np::from_object(result[1], np::dtype::get_builtin<float>(), 1);
		assert(static_cast<std::size_t>(behaviour_policy.shape(0)) == batch_size);
		assert(behaviour_policy.strides(0) == sizeof(float));
		std::vector<std::tuple<std::int64_t, float>> data;
		data.reserve(batch_size);
		for (auto i : ranges::view::indices(batch_size)) {
			auto action_id = reinterpret_cast<std::int64_t*>(actions.get_data())[i];
			auto bp = reinterpret_cast<float*>(behaviour_policy.get_data())[i];
			data.emplace_back(action_id, bp);
		}
		return data;
	} catch (boost::python::error_already_set) {
		::PyErr_Print();
		std::terminate();
	}
}

Network::Loss Network::train(ranges::span<typename StateTraits::value_type> states, ranges::span<std::int64_t> action_ids, ranges::span<Reward> rewards, ranges::span<float> behaviour_policies, ranges::span<std::int64_t> data_sizes)
{
	try {
		namespace np = boost::python::numpy;
		const auto t_max = static_cast<std::size_t>(data_sizes.size() - 1);
		const auto batch_size = static_cast<std::size_t>(states.size()) / StateTraits::size_of_all / (t_max + 1);
		assert(static_cast<std::size_t>(action_ids.size()) == batch_size * t_max && static_cast<std::size_t>(rewards.size()) == batch_size * t_max && static_cast<std::size_t>(behaviour_policies.size()) == batch_size * t_max);
		auto states_ndarray = StateTraits::convertToBatchedNdArray(states, t_max + 1, batch_size);
		auto action_ids_ndarray = NdArrayTraits<std::int64_t, 1>::convertToBatchedNdArray(action_ids, t_max, batch_size);
		auto rewards_ndarray = NdArrayTraits<Reward, 1>::convertToBatchedNdArray(rewards, t_max, batch_size);
		auto bp_ndarray = NdArrayTraits<float, 1>::convertToBatchedNdArray(behaviour_policies, t_max, batch_size);
		boost::python::list data_sizes_list;
		for (auto&& s : data_sizes) {
			data_sizes_list.append(s);
		}
		auto result = m_train_func(states_ndarray, action_ids_ndarray, rewards_ndarray, bp_ndarray, data_sizes_list);
		Loss loss;
		loss.v_loss = boost::python::extract<double>(result[0]);
		loss.pi_loss = boost::python::extract<double>(result[1]);
		loss.entropy_loss = boost::python::extract<double>(result[2]);
		return loss;
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
