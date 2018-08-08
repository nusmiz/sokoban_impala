#include <fstream>
#include <iostream>

#include "sokoban_env.hpp"

namespace impala
{


using ImageData = std::array<std::array<std::array<float, 8>, 8>, 3>;

namespace sokoban_image
{

// clang-format off
constexpr ImageData EMPTY = {{
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}},
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}},
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}}
}};

constexpr ImageData WALL = {{
	{{
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.6f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f}},
		{{0.6f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f}},
		{{0.6f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.5f,0.5f,0.5f,0.5f,0.6f,0.5f,0.5f,0.5f}},
		{{0.5f,0.5f,0.5f,0.5f,0.6f,0.5f,0.5f,0.5f}},
		{{0.5f,0.5f,0.5f,0.5f,0.6f,0.5f,0.5f,0.5f}}
	}},
	{{
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.6f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.6f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.6f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.0f,0.0f,0.0f,0.0f,0.6f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.6f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.6f,0.0f,0.0f,0.0f}}
	}},
	{{
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.6f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.6f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.6f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.0f,0.0f,0.0f,0.0f,0.6f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.6f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.6f,0.0f,0.0f,0.0f}}
	}}
}};

constexpr ImageData PLAYER = {{
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}},
	{{
		{{0.0f,0.0f,0.6f,0.0f,0.0f,0.6f,0.0f,0.0f}},
		{{0.0f,0.0f,0.6f,0.6f,0.6f,0.6f,0.0f,0.0f}},
		{{0.0f,0.6f,0.0f,0.6f,0.6f,0.0f,0.6f,0.0f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.0f,0.6f,0.0f,0.0f,0.0f,0.0f,0.6f,0.0f}},
		{{0.0f,0.6f,0.0f,0.0f,0.0f,0.0f,0.6f,0.0f}}
	}},
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}}
}};

constexpr ImageData BOX = {{
	{{
		{{0.64f,0.64f,0.64f,0.64f,0.64f,0.64f,0.64f,0.64f}},
		{{0.64f,0.64f,0.90f,0.90f,0.90f,0.90f,0.64f,0.64f}},
		{{0.64f,0.90f,0.64f,0.90f,0.90f,0.64f,0.90f,0.64f}},
		{{0.64f,0.90f,0.90f,0.64f,0.64f,0.90f,0.90f,0.64f}},
		{{0.64f,0.90f,0.90f,0.64f,0.64f,0.90f,0.90f,0.64f}},
		{{0.64f,0.90f,0.64f,0.90f,0.90f,0.64f,0.90f,0.64f}},
		{{0.64f,0.64f,0.90f,0.90f,0.90f,0.90f,0.64f,0.64f}},
		{{0.64f,0.64f,0.64f,0.64f,0.64f,0.64f,0.64f,0.64f}}
	}},
	{{
		{{0.33f,0.33f,0.33f,0.33f,0.33f,0.33f,0.33f,0.33f}},
		{{0.33f,0.33f,0.61f,0.61f,0.61f,0.61f,0.33f,0.33f}},
		{{0.33f,0.61f,0.33f,0.61f,0.61f,0.33f,0.61f,0.33f}},
		{{0.33f,0.61f,0.61f,0.33f,0.33f,0.61f,0.61f,0.33f}},
		{{0.33f,0.61f,0.61f,0.33f,0.33f,0.61f,0.61f,0.33f}},
		{{0.33f,0.61f,0.33f,0.61f,0.61f,0.33f,0.61f,0.33f}},
		{{0.33f,0.33f,0.61f,0.61f,0.61f,0.61f,0.33f,0.33f}},
		{{0.33f,0.33f,0.33f,0.33f,0.33f,0.33f,0.33f,0.33f}}
	}},
	{{
		{{0.05f,0.05f,0.05f,0.05f,0.05f,0.05f,0.05f,0.05f}},
		{{0.05f,0.05f,0.21f,0.21f,0.21f,0.21f,0.05f,0.05f}},
		{{0.05f,0.21f,0.05f,0.21f,0.21f,0.05f,0.21f,0.05f}},
		{{0.05f,0.21f,0.21f,0.05f,0.05f,0.21f,0.21f,0.05f}},
		{{0.05f,0.21f,0.21f,0.05f,0.05f,0.21f,0.21f,0.05f}},
		{{0.05f,0.21f,0.05f,0.21f,0.21f,0.05f,0.21f,0.05f}},
		{{0.05f,0.05f,0.21f,0.21f,0.21f,0.21f,0.05f,0.05f}},
		{{0.05f,0.05f,0.05f,0.05f,0.05f,0.05f,0.05f,0.05f}}
	}}
}};

constexpr ImageData TARGET = {{
	{{
		{{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}},
		{{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f}},
		{{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f}},
		{{1.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,1.0f}},
		{{1.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,1.0f}},
		{{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f}},
		{{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f}},
		{{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f}}
	}},
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}},
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}}
}};

constexpr ImageData PLAYER_TARGET = {{
	{{
		{{1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f}},
		{{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f}},
		{{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f}},
		{{1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f}}
	}},
	{{
		{{0.0f,0.0f,0.6f,0.0f,0.0f,0.6f,0.0f,0.0f}},
		{{0.0f,0.0f,0.6f,0.6f,0.6f,0.6f,0.0f,0.0f}},
		{{0.0f,0.6f,0.0f,0.6f,0.6f,0.0f,0.6f,0.0f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f}},
		{{0.0f,0.6f,0.0f,0.0f,0.0f,0.0f,0.6f,0.0f}},
		{{0.0f,0.6f,0.0f,0.0f,0.0f,0.0f,0.6f,0.0f}}
	}},
	{{
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}},
		{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}}
	}}
}};

constexpr ImageData BOX_TARGET = {{
	{{
		{{1.00f,1.00f,1.00f,1.00f,1.00f,1.00f,1.00f,1.00f}},
		{{1.00f,1.00f,0.90f,0.90f,0.90f,0.90f,1.00f,1.00f}},
		{{1.00f,0.90f,0.64f,0.90f,0.90f,0.64f,0.90f,1.00f}},
		{{1.00f,0.90f,0.90f,0.64f,0.64f,0.90f,0.90f,1.00f}},
		{{1.00f,0.90f,0.90f,0.64f,0.64f,0.90f,0.90f,1.00f}},
		{{1.00f,0.90f,0.64f,0.90f,0.90f,0.64f,0.90f,1.00f}},
		{{1.00f,1.00f,0.90f,0.90f,0.90f,0.90f,1.00f,1.00f}},
		{{1.00f,1.00f,1.00f,1.00f,1.00f,1.00f,1.00f,1.00f}},
	}},
	{{
		{{0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f}},
		{{0.00f,0.00f,0.61f,0.61f,0.61f,0.61f,0.00f,0.00f}},
		{{0.00f,0.61f,0.33f,0.61f,0.61f,0.33f,0.61f,0.00f}},
		{{0.00f,0.61f,0.61f,0.33f,0.33f,0.61f,0.61f,0.00f}},
		{{0.00f,0.61f,0.61f,0.33f,0.33f,0.61f,0.61f,0.00f}},
		{{0.00f,0.61f,0.33f,0.61f,0.61f,0.33f,0.61f,0.00f}},
		{{0.00f,0.00f,0.61f,0.61f,0.61f,0.61f,0.00f,0.00f}},
		{{0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f}}
	}},
	{{
		{{0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f}},
		{{0.00f,0.00f,0.21f,0.21f,0.21f,0.21f,0.00f,0.00f}},
		{{0.00f,0.21f,0.05f,0.21f,0.21f,0.05f,0.21f,0.00f}},
		{{0.00f,0.21f,0.21f,0.05f,0.05f,0.21f,0.21f,0.00f}},
		{{0.00f,0.21f,0.21f,0.05f,0.05f,0.21f,0.21f,0.00f}},
		{{0.00f,0.21f,0.05f,0.21f,0.21f,0.05f,0.21f,0.00f}},
		{{0.00f,0.00f,0.21f,0.21f,0.21f,0.21f,0.00f,0.00f}},
		{{0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f,0.00f}}
	}}
}};

// clang-format on

constexpr std::array<ImageData, 7> images = {{EMPTY, WALL, PLAYER, BOX, TARGET, PLAYER_TARGET, BOX_TARGET}};

}  // namespace sokoban_image

std::tuple<SokobanEnv::Observation, SokobanEnv::Reward, EnvState> SokobanEnv::step(const Action& action)
{
	int player_x;
	int player_y;
	bool found = [&]() -> bool {
		for (int y = 0; y < ROOM_HEIGHT; ++y) {
			for (int x = 0; x < ROOM_WIDTH; ++x) {
				if (m_states[y][x] == CellState::PLAYER || m_states[y][x] == CellState::PLAYER_TARGET) {
					player_x = x;
					player_y = y;
					return true;
				}
			}
		}
		return false;
	}();
	if (!found) {
		std::cerr << "cannot found player" << std::endl;
		std::terminate();
	}
	int diff_x;
	int diff_y;
	if (action == FourDirections::LEFT) {
		diff_x = -1;
		diff_y = 0;
	} else if (action == FourDirections::RIGHT) {
		diff_x = 1;
		diff_y = 0;
	} else if (action == FourDirections::UP) {
		diff_x = 0;
		diff_y = -1;
	} else if (action == FourDirections::DOWN) {
		diff_x = 0;
		diff_y = 1;
	} else {
		assert(false);
		diff_x = 0;
		diff_y = 0;
	}
	float reward = -0.1f;
	if (player_x + diff_x < 0 || player_x + diff_x >= ROOM_WIDTH || player_y + diff_y < 0 || player_y + diff_y >= ROOM_HEIGHT) {
		return std::make_tuple(Observation::copy(m_states), reward, EnvState::RUNNING);
	}
	auto& player_cell = m_states[player_y][player_x];
	auto& target = m_states[player_y + diff_y][player_x + diff_x];
	if (target == CellState::EMPTY) {
		if (player_cell == CellState::PLAYER_TARGET) {
			player_cell = CellState::TARGET;
		} else {
			player_cell = CellState::EMPTY;
		}
		target = CellState::PLAYER;
	} else if (target == CellState::TARGET) {
		if (player_cell == CellState::PLAYER_TARGET) {
			player_cell = CellState::TARGET;
		} else {
			player_cell = CellState::EMPTY;
		}
		target = CellState::PLAYER_TARGET;
	} else if (target == CellState::BOX) {
		if (player_x + diff_x * 2 < 0 || player_x + diff_x * 2 >= ROOM_WIDTH || player_y + diff_y * 2 < 0 || player_y + diff_y * 2 >= ROOM_HEIGHT) {
			return std::make_tuple(Observation::copy(m_states), reward, EnvState::RUNNING);
		}
		auto& target_next = m_states[player_y + diff_y * 2][player_x + diff_x * 2];
		if (target_next == CellState::EMPTY) {
			target_next = CellState::BOX;
			target = CellState::PLAYER;
			if (player_cell == CellState::PLAYER_TARGET) {
				player_cell = CellState::TARGET;
			} else {
				player_cell = CellState::EMPTY;
			}
		} else if (target_next == CellState::TARGET) {
			target_next = CellState::BOX_TARGET;
			target = CellState::PLAYER;
			if (player_cell == CellState::PLAYER_TARGET) {
				player_cell = CellState::TARGET;
			} else {
				player_cell = CellState::EMPTY;
			}
			reward += 1.0f;
		}
	} else if (target == CellState::BOX_TARGET) {
		if (player_x + diff_x * 2 < 0 || player_x + diff_x * 2 >= ROOM_WIDTH || player_y + diff_y * 2 < 0 || player_y + diff_y * 2 >= ROOM_HEIGHT) {
			return std::make_tuple(Observation::copy(m_states), reward, EnvState::RUNNING);
		}
		auto& target_next = m_states[player_y + diff_y * 2][player_x + diff_x * 2];
		if (target_next == CellState::EMPTY) {
			target_next = CellState::BOX;
			target = CellState::PLAYER_TARGET;
			if (player_cell == CellState::PLAYER_TARGET) {
				player_cell = CellState::TARGET;
			} else {
				player_cell = CellState::EMPTY;
			}
			reward -= 1.0f;
		} else if (target_next == CellState::TARGET) {
			target_next = CellState::BOX_TARGET;
			target = CellState::PLAYER_TARGET;
			if (player_cell == CellState::PLAYER_TARGET) {
				player_cell = CellState::TARGET;
			} else {
				player_cell = CellState::EMPTY;
			}
		}
	}
	bool done = [&]() -> bool {
		for (int y = 0; y < ROOM_HEIGHT; ++y) {
			for (int x = 0; x < ROOM_WIDTH; ++x) {
				if (m_states[y][x] == CellState::BOX) {
					return false;
				}
			}
		}
		return true;
	}();
	if (done) {
		reward += 10.0f;
	}
	return std::make_tuple(Observation::copy(m_states), reward, done ? EnvState::FINISHED : EnvState::RUNNING);
}


void SokobanEnv::writeData(const Observation& obs, TensorRef<float, 3, IMAGE_HEIGHT, IMAGE_WIDTH>& dest)
{
	auto copy_image = [&](int x, int y, CellState state) {
		auto& image = sokoban_image::images[static_cast<std::size_t>(state)];
		for (int dy = 0; dy < 8; ++dy) {
			for (int dx = 0; dx < 8; ++dx) {
				dest[0][y * 8 + dy][x * 8 + dx] = image[0][dy][dx];
			}
		}
		for (int dy = 0; dy < 8; ++dy) {
			for (int dx = 0; dx < 8; ++dx) {
				dest[1][y * 8 + dy][x * 8 + dx] = image[1][dy][dx];
			}
		}
		for (int dy = 0; dy < 8; ++dy) {
			for (int dx = 0; dx < 8; ++dx) {
				dest[2][y * 8 + dy][x * 8 + dx] = image[2][dy][dx];
			}
		}
	};
	for (int x = 0; x < ROOM_WIDTH + BORDER_WIDTH * 2; ++x) {
		for (int b = 0; b < BORDER_WIDTH; ++b) {
			copy_image(x, b, CellState::WALL);
			copy_image(x, ROOM_HEIGHT + BORDER_WIDTH + b, CellState::WALL);
		}
	}
	for (int y = BORDER_WIDTH; y < ROOM_HEIGHT + BORDER_WIDTH; ++y) {
		for (int b = 0; b < BORDER_WIDTH; ++b) {
			copy_image(b, y, CellState::WALL);
			copy_image(ROOM_WIDTH + BORDER_WIDTH + b, y, CellState::WALL);
		}
	}
	for (int y = 0; y < ROOM_HEIGHT; ++y) {
		for (int x = 0; x < ROOM_WIDTH; ++x) {
			copy_image(x + BORDER_WIDTH, y + BORDER_WIDTH, obs[y][x]);
		}
	}
}

void SokobanEnv::loadProblems()
{
	m_problems.clear();
	std::ifstream in{"./sokoban_problems.txt"};
	[&] {
		while (in) {
			Observation obs;
			for (int y = 0; y < ROOM_HEIGHT; ++y) {
				for (int x = 0; x < ROOM_WIDTH; ++x) {
					int data;
					in >> data;
					if (!in) {
						return;
					}
					assert(0 <= data && data < 7);
					obs[y][x] = static_cast<CellState>(data);
				}
			}
			m_problems.emplace_back(std::move(obs));
		}
	}();
	m_problems.shrink_to_fit();
	std::cout << "load " << m_problems.size() << " problems" << std::endl;
}

std::vector<SokobanEnv::Observation> SokobanEnv::m_problems;


}  // namespace impala
