#include <iostream>
#include <memory>

#include "action.hpp"
#include "environment.hpp"
#include "network.hpp"
#include "python_util.hpp"
#include "server.hpp"
#include "sokoban_env.hpp"
#include "tensor.hpp"

int main()
{
	using namespace impala;
	PythonInitializer py_initializer{false};
	SokobanEnv::loadProblems();
	auto server = std::make_unique<Server<SokobanEnv, Network, 2048>>(120);
	server->run(1000000000);
	return 0;
}
