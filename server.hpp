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
#include <range/v3/view/zip.hpp>

#include "environment.hpp"

namespace impala
{

struct DefaultServerParams
{
	static inline constexpr std::size_t NUM_AGENTS = 2048;
	static inline constexpr std::size_t NUM_PREDICTORS = 2;
	static inline constexpr std::size_t NUM_TRAINERS = 2;

	static inline constexpr std::size_t MIN_PREDICTION_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_PREDICTION_BATCH_SIZE = 1024;
	static inline constexpr std::size_t MIN_TRAINING_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_TRAINING_BATCH_SIZE = 1024;

	static inline constexpr std::size_t T_MAX = 5;
	static inline constexpr std::optional<std::size_t> MAX_EPISODE_LENGTH = std::nullopt;

	static inline constexpr std::optional<std::size_t> LOG_INTERVAL_STEPS = 10000;
	static inline constexpr std::optional<std::size_t> SAVE_INTERVAL_STEPS = 1000000;
};

template <class Environment, class Model, class Parameters = DefaultServerParams>
class Server
{
public:
	static_assert(IsEnvironmentV<Environment>);

	using Reward = typename Environment::Reward;
	using Observation = typename Environment::Observation;
	using ObsBatch = typename Environment::ObsBatch;
	using Action = typename Environment::Action;

	static inline constexpr std::size_t NUM_AGENTS = Parameters::NUM_AGENTS;
	static inline constexpr std::size_t NUM_PREDICTORS = Parameters::NUM_PREDICTORS;
	static inline constexpr std::size_t NUM_TRAINERS = Parameters::NUM_TRAINERS;

	static inline constexpr std::size_t MIN_PREDICTION_BATCH_SIZE = Parameters::MIN_PREDICTION_BATCH_SIZE;
	static inline constexpr std::size_t MAX_PREDICTION_BATCH_SIZE = Parameters::MAX_PREDICTION_BATCH_SIZE;
	static inline constexpr std::size_t MIN_TRAINING_BATCH_SIZE = Parameters::MIN_TRAINING_BATCH_SIZE;
	static inline constexpr std::size_t MAX_TRAINING_BATCH_SIZE = Parameters::MAX_TRAINING_BATCH_SIZE;

	static inline constexpr std::size_t T_MAX = Parameters::T_MAX;
	static inline constexpr std::optional<std::size_t> MAX_EPISODE_LENGTH = Parameters::MAX_EPISODE_LENGTH;

	static inline constexpr std::optional<std::size_t> LOG_INTERVAL_STEPS = Parameters::LOG_INTERVAL_STEPS;
	static inline constexpr std::optional<std::size_t> SAVE_INTERVAL_STEPS = Parameters::SAVE_INTERVAL_STEPS;

	Server()
	{
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_PREDICTORS)) {
			m_predictors.emplace_back(*this);
		}
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_TRAINERS)) {
			m_trainers.emplace_back(*this);
		}
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_AGENTS)) {
			m_agents.emplace_back(*this);
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
				auto [v_loss, pi_loss, entropy_loss] = m_model.train(batch.states, batch.actions, batch.rewards, batch.policies, batch.data_sizes, batch.observation_sizes);
				batch.trainer.get().processFinished();
				average_v_loss = average_loss_decay * average_v_loss + (1.0 - average_loss_decay) * v_loss;
				average_pi_loss = average_loss_decay * average_pi_loss + (1.0 - average_loss_decay) * pi_loss;
				average_entropy_loss = average_loss_decay * average_entropy_loss + (1.0 - average_loss_decay) * entropy_loss;
				auto prev_trained_steps = trained_steps;
				for (auto i : ranges::view::indices(T_MAX)) {
					trained_steps += batch.data_sizes.at(i);
				}
				if constexpr (LOG_INTERVAL_STEPS.has_value()) {
					if (trained_steps / LOG_INTERVAL_STEPS.value() != prev_trained_steps / LOG_INTERVAL_STEPS.value()) {
						std::cout << "steps " << trained_steps << " , loss " << average_v_loss << " " << average_pi_loss << " " << average_entropy_loss << std::endl;
					}
				}
				if constexpr (SAVE_INTERVAL_STEPS.has_value()) {
					if (trained_steps / SAVE_INTERVAL_STEPS.value() != prev_trained_steps / SAVE_INTERVAL_STEPS.value()) {
						m_model.save(static_cast<int>(trained_steps));
					}
				}
			}
			for (auto&& batch : prediction_batches) {
				auto actions_and_policies = m_model.predict(batch.states);
				assert(actions_and_policies.size() == batch.agents.size());
				batch.predictor.get().processFinished();
				for (auto&& [agent, action_and_policy] : ranges::view::zip(batch.agents, actions_and_policies)) {
					auto&& [action, policy] = action_and_policy;
					agent.get().setNextActionAndPolicy(DiscreteActionTraits<Action>::convertFromID(action), policy);
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

	struct PredictionData
	{
		std::reference_wrapper<std::add_const_t<Observation>> observation;
		std::reference_wrapper<Agent> agent;
	};
	struct PredictionBatch
	{
		ObsBatch states;
		std::vector<std::reference_wrapper<Agent>> agents;
		std::reference_wrapper<Predictor> predictor;
	};
	struct TrainingData
	{
		std::vector<Observation> observations;
		std::vector<Action> actions;
		std::vector<Reward> rewards;
		std::vector<float> policies;
	};
	struct TrainingBatch
	{
		std::array<std::int64_t, T_MAX> data_sizes;
		std::array<std::int64_t, T_MAX + 1> observation_sizes;
		ObsBatch states;
		std::vector<std::int64_t> actions;
		std::vector<Reward> rewards;
		std::vector<float> policies;
		std::reference_wrapper<Trainer> trainer;
	};

	class Predictor
	{
	public:
		explicit Predictor(Server& server) noexcept : m_server(server)
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
				std::vector<std::reference_wrapper<Agent>> agents;
				observations.reserve(MAX_PREDICTION_BATCH_SIZE);
				agents.reserve(MAX_PREDICTION_BATCH_SIZE);
				bool data_remain = false;
				{
					std::unique_lock lock{m_server.get().m_prediction_queue_lock};
					m_server.get().m_predictor_event.wait(lock, [this] { return m_server.get().m_prediction_queue.size() >= MIN_PREDICTION_BATCH_SIZE || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
					auto& queue = m_server.get().m_prediction_queue;
					while (!queue.empty()) {
						if (observations.size() >= MAX_PREDICTION_BATCH_SIZE) {
							break;
						}
						auto& data = queue.front();
						observations.emplace_back(data.observation);
						agents.emplace_back(data.agent);
						queue.pop_front();
					}
					data_remain = (queue.size() >= MIN_PREDICTION_BATCH_SIZE);
				}
				if (data_remain) {
					m_server.get().m_predictor_event.notify_one();
				}
				PredictionBatch batch{Environment::makeBatch(observations.begin(), observations.end()), std::move(agents), *this};
				{
					std::lock_guard lock{m_server.get().m_batches_lock};
					m_server.get().m_prediction_batches.emplace_back(std::move(batch));
					m_processing_flag = true;
				}
				m_server.get().m_server_event.notify_one();
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
		std::reference_wrapper<Server> m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		bool m_processing_flag = false;
		bool m_exit_flag = false;
	};

	class Trainer
	{
	public:
		explicit Trainer(Server& server) noexcept : m_server(server)
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
			std::vector<TrainingData> datas;
			datas.reserve(MAX_TRAINING_BATCH_SIZE);
			std::vector<std::optional<Observation>> observations;
			observations.reserve(MAX_TRAINING_BATCH_SIZE * (T_MAX + 1));
			while (true) {
				datas.clear();
				observations.clear();
				std::vector<std::int64_t> actions;
				std::vector<Reward> rewards;
				std::vector<float> policies;
				actions.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
				rewards.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
				policies.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
				bool data_remain = false;
				{
					std::unique_lock lock{m_server.get().m_training_queue_lock};
					m_server.get().m_trainer_event.wait(lock, [this] { return m_server.get().m_training_queue.size() >= MIN_TRAINING_BATCH_SIZE || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
					auto& queue = m_server.get().m_training_queue;
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
					m_server.get().m_trainer_event.notify_one();
				}
				std::sort(datas.begin(), datas.end(), [](const auto& a, const auto& b) {
					if (a.actions.size() == b.actions.size()) {
						return a.observations.size() > b.observations.size();
					}
					return a.actions.size() > b.actions.size();
				});
				for (auto i : ranges::view::indices(T_MAX)) {
					for (auto& data : datas) {
						if (i >= data.actions.size()) {
							if (i >= data.observations.size()) {
								observations.emplace_back(std::nullopt);
							} else {
								observations.emplace_back(std::move(data.observations.at(i)));
							}
							actions.emplace_back(0);
							rewards.emplace_back(Reward{});
							policies.emplace_back(0.0f);
						} else {
							observations.emplace_back(std::move(data.observations.at(i)));
							actions.emplace_back(DiscreteActionTraits<Action>::convertToID(data.actions.at(i)));
							rewards.emplace_back(std::move(data.rewards.at(i)));
							policies.emplace_back(std::move(data.policies.at(i)));
						}
					}
				}
				for (auto& data : datas) {
					if (data.observations.size() >= T_MAX + 1) {
						observations.emplace_back(std::move(data.observations.back()));
					} else {
						observations.emplace_back(std::nullopt);
					}
				}

				TrainingBatch batch{{}, {}, Environment::makeBatch(observations.cbegin(), observations.cend()), std::move(actions), std::move(rewards), std::move(policies), *this};
				for (auto i : ranges::view::indices(T_MAX)) {
					auto it = std::upper_bound(datas.begin(), datas.end(), i + 1, [](std::size_t x, const auto& y) {
						return x > y.actions.size();
					});
					batch.data_sizes.at(i) = it - datas.begin();
				}
				for (auto i : ranges::view::indices(T_MAX + 1)) {
					auto it = std::upper_bound(datas.begin(), datas.end(), i + 1, [](std::size_t x, const auto& y) {
						return x > y.observations.size();
					});
					batch.observation_sizes.at(i) = it - datas.begin();
				}
				{
					std::lock_guard lock{m_server.get().m_batches_lock};
					m_server.get().m_training_batches.emplace_back(std::move(batch));
					m_processing_flag = true;
				}
				m_server.get().m_server_event.notify_one();
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
		std::reference_wrapper<Server> m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		bool m_processing_flag = false;
		bool m_exit_flag = false;
	};

	class Agent
	{
	public:
		explicit Agent(Server& server) noexcept : m_server(server)
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
			std::vector<float> prev_policies;
			prev_obss.reserve(T_MAX + 1);
			prev_actions.reserve(T_MAX + 1);
			prev_rewards.reserve(T_MAX + 1);
			prev_policies.reserve(T_MAX + 1);
			while (true) {
				prev_obss.clear();
				prev_actions.clear();
				prev_rewards.clear();
				prev_policies.clear();
				Reward sum_of_reward = Reward{};
				std::size_t t = 0;
				Observation observation = m_env.reset();
				while (true) {
					if constexpr (MAX_EPISODE_LENGTH.has_value()) {
						if (t >= MAX_EPISODE_LENGTH.value()) {
							break;
						}
					}
					{
						bool enough_predictor_data = false;
						{
							std::lock_guard lock{m_server.get().m_prediction_queue_lock};
							m_server.get().m_prediction_queue.emplace_back(PredictionData{std::cref(observation), *this});
							m_predicting_flag = true;
							enough_predictor_data = m_server.get().m_prediction_queue.size() >= MIN_PREDICTION_BATCH_SIZE;
						}
						if (enough_predictor_data) {
							m_server.get().m_predictor_event.notify_one();
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
					++t;
					sum_of_reward += current_reward;
					if (status == EnvState::FINISHED || prev_obss.size() >= T_MAX || (MAX_EPISODE_LENGTH.has_value() && t >= MAX_EPISODE_LENGTH.value())) {
						assert(prev_obss.size() == prev_actions.size() && prev_obss.size() == prev_rewards.size());
						TrainingData data;
						std::optional<TrainingData> data2;
						for (auto&& [obs, action, reward, policy] : ranges::view::zip(prev_obss, prev_actions, prev_rewards, prev_policies)) {
							data.observations.emplace_back(std::move(obs));
							data.actions.emplace_back(std::move(action));
							data.rewards.emplace_back(std::move(reward));
							data.policies.emplace_back(std::move(policy));
						}
						if (status == EnvState::FINISHED) {
							if (data.actions.size() < T_MAX) {
								data.observations.emplace_back(std::move(observation));
								data.actions.emplace_back(std::move(next_action));
								data.rewards.emplace_back(std::move(current_reward));
								data.policies.emplace_back(std::move(policy));
							} else {
								data.observations.emplace_back(observation.clone());
								data2.emplace();
								data2->observations.emplace_back(std::move(observation));
								data2->actions.emplace_back(std::move(next_action));
								data2->rewards.emplace_back(std::move(current_reward));
								data2->policies.emplace_back(std::move(policy));
							}
						} else {
							data.observations.emplace_back(observation.clone());
						}
						bool enough_trainer_data = false;
						{
							std::lock_guard lock{m_server.get().m_training_queue_lock};
							auto& queue = m_server.get().m_training_queue;
							queue.emplace_back(std::move(data));
							if (data2) {
								queue.emplace_back(std::move(data2.value()));
							}
							enough_trainer_data = (queue.size() >= MIN_TRAINING_BATCH_SIZE);
						}
						if (enough_trainer_data) {
							m_server.get().m_trainer_event.notify_one();
						}
						prev_obss.clear();
						prev_actions.clear();
						prev_rewards.clear();
						prev_policies.clear();
						if (status == EnvState::FINISHED) {
							break;
						}
					}
					prev_obss.emplace_back(std::move(observation));
					observation = std::move(next_obs);
					prev_actions.emplace_back(std::move(next_action));
					prev_rewards.emplace_back(std::move(current_reward));
					prev_policies.emplace_back(std::move(policy));
				}
				if (this == &m_server.get().m_agents.front()) {
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
		std::reference_wrapper<Server> m_server;
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
	std::deque<PredictionData> m_prediction_queue;
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
