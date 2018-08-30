#include <iostream>
#include <memory>

#include "action.hpp"
#include "environment.hpp"
#include "network.hpp"
#include "python_util.hpp"
#include "server.hpp"
#include "sokoban_env.hpp"
#include "tensor.hpp"

struct SokobanTrainParams
{
	static inline constexpr std::size_t NUM_AGENTS = 2048;
	static inline constexpr std::size_t NUM_PREDICTORS = 2;
	static inline constexpr std::size_t NUM_TRAINERS = 2;

	static inline constexpr std::size_t MIN_PREDICTION_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_PREDICTION_BATCH_SIZE = 1024;
	static inline constexpr std::size_t MIN_TRAINING_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_TRAINING_BATCH_SIZE = 1024;

	static inline constexpr std::size_t T_MAX = 5;
	static inline constexpr std::optional<std::size_t> MAX_EPISODE_LENGTH = 120;

	static inline constexpr std::optional<std::size_t> LOG_INTERVAL_STEPS = 10000;
	static inline constexpr std::optional<std::size_t> SAVE_INTERVAL_STEPS = 1000000;
};

int main()
{
	using namespace impala;
	PythonInitializer py_initializer{false};
	SokobanEnv::loadProblems();
	auto server = std::make_unique<Server<SokobanEnv, Network, SokobanTrainParams>>();
	server->run(1000000000);
	return 0;
}
