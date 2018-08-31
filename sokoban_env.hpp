#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include "action.hpp"
#include "environment.hpp"
#include "python_util.hpp"
#include "tensor.hpp"

namespace impala
{

class SokobanEnv
{
public:
	enum class CellState : std::uint8_t
	{
		EMPTY,
		WALL,
		PLAYER,
		BOX,
		TARGET,
		PLAYER_TARGET,
		BOX_TARGET
	};

	static constexpr int ROOM_WIDTH = 8;
	static constexpr int ROOM_HEIGHT = 8;
	static constexpr int BORDER_WIDTH = 1;
	static constexpr int IMAGE_WIDTH = 8 * (ROOM_WIDTH + BORDER_WIDTH * 2);
	static constexpr int IMAGE_HEIGHT = 8 * (ROOM_HEIGHT + BORDER_WIDTH * 2);

	using Observation = Tensor<CellState, ROOM_HEIGHT, ROOM_WIDTH>;
	using ObsBatch = std::vector<float>;
	using Reward = float;
	using Action = FourDirections;

	SokobanEnv() : m_random_engine{std::random_device{}()} {}

	Observation reset()
	{
		auto index = std::uniform_int_distribution<std::size_t>{0, m_problems.size() - 1}(m_random_engine);
		m_states = m_problems.at(index).clone();
		return m_states.clone();
	}
	std::tuple<Observation, Reward, EnvState> step(const Action& action);
	void render() const {}

	template <class ForwardIterator,
	    std::enable_if_t<
	        std::conjunction_v<
	            std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<ForwardIterator>::iterator_category>,
	            std::disjunction<
	                std::is_convertible<typename std::iterator_traits<ForwardIterator>::reference, const Observation&>,
	                std::is_convertible<typename std::iterator_traits<ForwardIterator>::reference, const std::optional<Observation>&>>>,
	        std::nullptr_t> = nullptr>
	static ObsBatch makeBatch(ForwardIterator first, ForwardIterator last)
	{
		return NdArrayTraits<float, 3, IMAGE_HEIGHT, IMAGE_WIDTH>::makeBufferForBatch(first, last, [](const auto& obs, auto& dest) {
			if constexpr (std::is_convertible_v<decltype(obs), const Observation&>) {
				writeData(obs, dest);
			} else {
				const std::optional<Observation>& opt = obs;
				if (opt.has_value()) {
					writeData(opt.value(), dest);
				}
			}
		});
	}

	static void loadProblems();

private:
	static void writeData(const Observation& obs, TensorRef<float, 3, IMAGE_HEIGHT, IMAGE_WIDTH>& dest);

	static std::vector<Observation> m_problems;

	Observation m_states;
	std::mt19937 m_random_engine;
};

static_assert(IsEnvironmentV<SokobanEnv>);

}  // namespace impala
