#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/zip.hpp>

#include "environment.hpp"

namespace impala
{

template <class Environment, class Model, std::size_t NUM_AGENTS, std::size_t NUM_PREDICTORS = 2, std::size_t NUM_TRAINERS = 2>
class Server
{
public:
	static_assert(IsEnvironmentV<Environment>);

	using Reward = typename Environment::Reward;
	using Observation = typename Environment::Observation;
	using ObsBatch = typename Environment::ObsBatch;
	using Action = typename Environment::Action;

	Server(int max_episode_length = -1) : m_max_episode_length{max_episode_length}
	{
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_PREDICTORS)) {
			m_predictors.emplace_back(this);
		}
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_TRAINERS)) {
			m_trainers.emplace_back(this);
		}
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_AGENTS)) {
			m_agents.emplace_back(this);
		}
	}
	~Server()
	{
		for (auto&& predictor : m_predictors) {
			predictor.exit();
		}
		m_predictor_event.notify_all();
		m_predictors.clear();
		for (auto&& trainer : m_trainers) {
			trainer.exit();
		}
		m_trainer_event.notify_all();
		m_trainers.clear();
		for (auto&& agent : m_agents) {
			agent.exit();
		}
		m_agents.clear();
	}

	void run(const std::size_t training_steps)
	{
		static constexpr double average_loss_decay = 0.99;

		std::size_t trained_steps = 0;

		double average_v_loss = 0;
		double average_pi_loss = 0;
		double average_entropy_loss = 0;

		std::vector<TrainingBatch> training_batches;
		std::vector<PredictionBatch> prediction_batches;
		while (true) {
			training_batches.clear();
			prediction_batches.clear();
			{
				std::unique_lock lock{m_batches_lock};
				m_server_event.wait(lock, [this] { return !m_training_batches.empty() || !m_prediction_batches.empty(); });
				std::swap(m_training_batches, training_batches);
				std::swap(m_prediction_batches, prediction_batches);
			}
			for (auto&& batch : training_batches) {
				auto states = Environment::batchToNetworkInput(batch.states);
				auto [v_loss, pi_loss, entropy_loss] = m_model.train(states, batch.actions, batch.rewards, batch.policy, batch.data_sizes);
				batch.trainer->processFinished();
				average_v_loss = average_loss_decay * average_v_loss + (1.0 - average_loss_decay) * v_loss;
				average_pi_loss = average_loss_decay * average_pi_loss + (1.0 - average_loss_decay) * pi_loss;
				average_entropy_loss = average_loss_decay * average_entropy_loss + (1.0 - average_loss_decay) * entropy_loss;
				auto prev_trained_steps = trained_steps;
				for (std::size_t i = 0; i < T_MAX; ++i) {
					trained_steps += batch.data_sizes.at(i);
				}
				if (trained_steps / 10000 != prev_trained_steps / 10000) {
					std::cout << "steps " << trained_steps << " , loss " << average_v_loss << " " << average_pi_loss << " " << average_entropy_loss << std::endl;
				}
				if (trained_steps / 1000000 != prev_trained_steps / 1000000) {
					m_model.save(static_cast<int>(trained_steps));
				}
			}
			for (auto&& batch : prediction_batches) {
				auto states = Environment::batchToNetworkInput(batch.states);
				auto actions_and_policy_list = m_model.predict(states);
				assert(actions_and_policy_list.size() == batch.agents.size());
				batch.predictor->processFinished();
				for (auto&& [agent, action_and_policy] : ranges::view::zip(batch.agents, actions_and_policy_list)) {
					auto&& [action, policy] = action_and_policy;
					agent->setNextActionAndPolicy(DiscreteActionTraits<Action>::convertFromID(action), policy);
				}
			}
			if (trained_steps >= training_steps) {
				std::cout << "training finished" << std::endl;
				break;
			}
		}
	}

private:
	class Predictor;
	class Trainer;
	class Agent;

	struct PredictionBatch
	{
		ObsBatch states;
		std::vector<Agent*> agents;
		Predictor* predictor;
	};
	struct TrainingData
	{
		std::vector<Observation> observations;
		std::vector<Action> actions;
		std::vector<Reward> rewards;
		std::vector<float> policy;
	};
	struct TrainingBatch
	{
		std::vector<std::int64_t> data_sizes;
		ObsBatch states;
		std::vector<std::int64_t> actions;
		std::vector<Reward> rewards;
		std::vector<float> policy;
		Trainer* trainer;
	};

	static inline constexpr std::size_t MIN_PREDICTION_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_PREDICTION_BATCH_SIZE = 1024;
	static inline constexpr std::size_t MIN_TRAINING_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_TRAINING_BATCH_SIZE = 1024;

	static inline constexpr int T_MAX = 5;

	class Predictor
	{
	public:
		explicit Predictor(Server* server) noexcept : m_server(server)
		{
			m_thread = std::thread{[this] {
				run();
			}};
		}
		~Predictor()
		{
			m_thread.join();
		}

		void run()
		{
			while (true) {
				std::vector<std::reference_wrapper<std::add_const_t<Observation>>> observations;
				std::vector<Agent*> agents;
				observations.reserve(MAX_PREDICTION_BATCH_SIZE);
				agents.reserve(MAX_PREDICTION_BATCH_SIZE);
				bool data_remain = false;
				{
					std::unique_lock lock{m_server->m_prediction_queue_lock};
					m_server->m_predictor_event.wait(lock, [this] { return m_server->m_prediction_queue.size() >= MIN_PREDICTION_BATCH_SIZE || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
					auto& queue = m_server->m_prediction_queue;
					while (!queue.empty()) {
						if (observations.size() >= MAX_PREDICTION_BATCH_SIZE) {
							break;
						}
						auto& data = queue.front();
						observations.emplace_back(std::move(std::get<0>(data)));
						agents.emplace_back(std::move(std::get<1>(data)));
						queue.pop_front();
					}
					data_remain = (queue.size() >= MIN_PREDICTION_BATCH_SIZE);
				}
				if (data_remain) {
					m_server->m_predictor_event.notify_one();
				}
				PredictionBatch batch;
				batch.states = Environment::makeBatch(observations.begin(), observations.end());
				batch.agents = std::move(agents);
				batch.predictor = this;
				{
					std::lock_guard lock{m_server->m_batches_lock};
					m_server->m_prediction_batches.emplace_back(std::move(batch));
					m_processing_flag = true;
				}
				m_server->m_server_event.notify_one();
				{
					std::unique_lock lock{m_mutex};
					m_event.wait(lock, [this] { return !m_processing_flag || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
				}
			}
		}

		void exit()
		{
			{
				std::lock_guard lock{m_mutex};
				m_exit_flag = true;
			}
			m_event.notify_one();
		}

		void processFinished()
		{
			{
				std::lock_guard lock{m_mutex};
				m_processing_flag = false;
			}
			m_event.notify_one();
		}

	private:
		Server* m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		bool m_processing_flag = false;
		bool m_exit_flag = false;
	};

	class Trainer
	{
	public:
		explicit Trainer(Server* server) noexcept : m_server(server)
		{
			m_thread = std::thread{[this] {
				run();
			}};
		}
		~Trainer()
		{
			m_thread.join();
		}

		void run()
		{
			while (true) {
				std::vector<TrainingData> datas;
				datas.reserve(MAX_TRAINING_BATCH_SIZE);
				std::vector<Observation> observations;
				std::vector<std::int64_t> actions;
				std::vector<Reward> rewards;
				std::vector<float> policy;
				observations.reserve(MAX_TRAINING_BATCH_SIZE * (T_MAX + 1));
				actions.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
				rewards.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
				policy.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
				bool data_remain = false;
				{
					std::unique_lock lock{m_server->m_training_queue_lock};
					m_server->m_trainer_event.wait(lock, [this] { return m_server->m_training_queue.size() >= MIN_TRAINING_BATCH_SIZE || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
					auto& queue = m_server->m_training_queue;
					while (!queue.empty()) {
						if (datas.size() >= MAX_TRAINING_BATCH_SIZE) {
							break;
						}
						auto& data = queue.front();
						datas.emplace_back(std::move(data));
						queue.pop_front();
					}
					data_remain = (queue.size() >= MIN_TRAINING_BATCH_SIZE);
				}
				if (data_remain) {
					m_server->m_trainer_event.notify_one();
				}
				std::sort(datas.begin(), datas.end(), [](const auto& a, const auto& b) {
					return a.observations.size() > b.observations.size();
				});
				for (std::size_t i = 0; i < T_MAX; ++i) {
					for (auto& data : datas) {
						if (i >= data.actions.size()) {
							observations.emplace_back(Observation{});
							actions.emplace_back(0);
							rewards.emplace_back(Reward{});
							policy.emplace_back(0.0f);
						} else {
							observations.emplace_back(std::move(data.observations.at(i)));
							actions.emplace_back(DiscreteActionTraits<Action>::convertToID(data.actions.at(i)));
							rewards.emplace_back(std::move(data.rewards.at(i)));
							policy.emplace_back(std::move(data.policy.at(i)));
						}
					}
				}
				for (auto& data : datas) {
					if (data.observations.size() >= T_MAX + 1) {
						observations.emplace_back(std::move(data.observations.back()));
					} else {
						observations.emplace_back(Observation{});
					}
				}

				TrainingBatch batch;
				batch.data_sizes.resize(T_MAX + 1);
				for (std::size_t i = 0; i < T_MAX + 1; ++i) {
					auto it = std::upper_bound(datas.begin(), datas.end(), i + 1, [](std::size_t x, const auto& y) {
    					return x > y.observations.size();
    				});
					batch.data_sizes.at(i) = it - datas.begin();
				}
				batch.states = Environment::makeBatch(observations.cbegin(), observations.cend());
				batch.actions = std::move(actions);
				batch.rewards = std::move(rewards);
				batch.policy = std::move(policy);
				batch.trainer = this;
				{
					std::lock_guard lock{m_server->m_batches_lock};
					m_server->m_training_batches.emplace_back(std::move(batch));
					m_processing_flag = true;
				}
				m_server->m_server_event.notify_one();
				{
					std::unique_lock lock{m_mutex};
					m_event.wait(lock, [this] { return !m_processing_flag || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
				}
			}
		}

		void exit()
		{
			{
				std::lock_guard lock{m_mutex};
				m_exit_flag = true;
			}
			m_event.notify_one();
		}

		void processFinished()
		{
			{
				std::lock_guard lock{m_mutex};
				m_processing_flag = false;
			}
			m_event.notify_one();
		}

	private:
		Server* m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		bool m_processing_flag = false;
		bool m_exit_flag = false;
	};

	class Agent
	{
	public:
		explicit Agent(Server* server) noexcept : m_server(server)
		{
			m_thread = std::thread{[this] {
				run();
			}};
		}
		~Agent()
		{
			m_thread.join();
		}

		void run()
		{
			std::vector<Observation> prev_obss;
			std::vector<Action> prev_actions;
			std::vector<Reward> prev_rewards;
			std::vector<float> prev_policy;
			prev_obss.reserve(T_MAX + 1);
			prev_actions.reserve(T_MAX + 1);
			prev_rewards.reserve(T_MAX + 1);
			prev_policy.reserve(T_MAX + 1);
			while (true) {
				prev_obss.clear();
				prev_actions.clear();
				prev_rewards.clear();
				prev_policy.clear();
				Reward sum_of_reward = Reward{};
				int t = 0;
				Observation observation = m_env.reset();
				for (; t != m_server->m_max_episode_length; ++t) {
					{
						bool enough_predictor_data = false;
						{
							std::lock_guard lock{m_server->m_prediction_queue_lock};
							m_server->m_prediction_queue.emplace_back(std::cref(observation), this);
							m_predicting_flag = true;
							enough_predictor_data = m_server->m_prediction_queue.size() >= MIN_PREDICTION_BATCH_SIZE;
						}
						if (enough_predictor_data) {
							m_server->m_predictor_event.notify_one();
						}
					}
					Action next_action;
					float policy;
					{
						std::unique_lock lock{m_mutex};
						m_event.wait(lock, [this] { return !m_predicting_flag || m_exit_flag; });
						if (m_exit_flag) {
							return;
						}
						next_action = m_action;
						policy = m_policy;
					}
					auto&& [next_obs, current_reward, status] = m_env.step(next_action);
					sum_of_reward += current_reward;
					if (status == EnvState::FINISHED || prev_obss.size() >= T_MAX) {
						assert(prev_obss.size() == prev_actions.size() && prev_obss.size() == prev_rewards.size());
						TrainingData data;
						std::optional<TrainingData> data2;
						for (auto&& [obs, action, reward, policy] : ranges::view::zip(prev_obss, prev_actions, prev_rewards, prev_policy)) {
							data.observations.emplace_back(std::move(obs));
							data.actions.emplace_back(std::move(action));
							data.rewards.emplace_back(std::move(reward));
							data.policy.emplace_back(std::move(policy));
						}
						if (status == EnvState::FINISHED) {
							if (data.actions.size() < T_MAX) {
								data.observations.emplace_back(std::move(observation));
								data.actions.emplace_back(std::move(next_action));
								data.rewards.emplace_back(std::move(current_reward));
								data.policy.emplace_back(std::move(policy));
							} else {
								data.observations.emplace_back(Observation::copy(observation));
								data2.emplace();
								data2->observations.emplace_back(std::move(observation));
								data2->actions.emplace_back(std::move(next_action));
								data2->rewards.emplace_back(std::move(current_reward));
								data2->policy.emplace_back(std::move(policy));
							}
						} else {
							data.observations.emplace_back(Observation::copy(observation));
						}
						bool enough_trainer_data = false;
						{
							std::lock_guard lock{m_server->m_training_queue_lock};
							auto& queue = m_server->m_training_queue;
							queue.emplace_back(std::move(data));
							if (data2) {
								queue.emplace_back(std::move(data2.value()));
							}
							enough_trainer_data = (queue.size() >= MIN_TRAINING_BATCH_SIZE);
						}
						if (enough_trainer_data) {
							m_server->m_trainer_event.notify_one();
						}
						prev_obss.clear();
						prev_actions.clear();
						prev_rewards.clear();
						prev_policy.clear();
						if (status == EnvState::FINISHED) {
							break;
						}
					}
					prev_obss.emplace_back(std::move(observation));
					observation = std::move(next_obs);
					prev_actions.emplace_back(std::move(next_action));
					prev_rewards.emplace_back(std::move(current_reward));
					prev_policy.emplace_back(std::move(policy));
				}
				if (this == &m_server->m_agents.front()) {
					std::cout << "finish episode : " << t << " " << std::setprecision(5) << sum_of_reward << std::endl;
				}
			}
		}

		void exit()
		{
			{
				std::lock_guard lock{m_mutex};
				m_exit_flag = true;
			}
			m_event.notify_one();
		}

		void setNextActionAndPolicy(Action action, float policy)
		{
			{
				std::lock_guard lock{m_mutex};
				m_action = action;
				m_policy = policy;
				m_predicting_flag = false;
			}
			m_event.notify_one();
		}

	private:
		Server* m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		Action m_action;
		float m_policy;
		bool m_predicting_flag = false;
		bool m_exit_flag = false;
		Environment m_env;
	};

	boost::container::static_vector<Predictor, NUM_PREDICTORS> m_predictors;
	boost::container::static_vector<Trainer, NUM_TRAINERS> m_trainers;
	boost::container::static_vector<Agent, NUM_AGENTS> m_agents;
	Model m_model;
	int m_max_episode_length = -1;
	std::deque<std::tuple<std::reference_wrapper<std::add_const_t<Observation>>, Agent*>> m_prediction_queue;
	std::mutex m_prediction_queue_lock;
	std::condition_variable m_predictor_event;
	std::deque<TrainingData> m_training_queue;
	std::mutex m_training_queue_lock;
	std::condition_variable m_trainer_event;
	std::vector<PredictionBatch> m_prediction_batches;
	std::vector<TrainingBatch> m_training_batches;
	std::mutex m_batches_lock;
	std::condition_variable m_server_event;
};

}  // namespace impala
