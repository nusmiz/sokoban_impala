#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>

namespace impala
{

template <class T>
struct DiscreteActionTraits;

namespace detail
{

template <class T,
    std::enable_if_t<
        std::conjunction_v<
            std::is_default_constructible<T>,
            std::is_same<decltype(DiscreteActionTraits<T>::num_actions), const std::int64_t>,
            std::is_same<std::int64_t, decltype(DiscreteActionTraits<T>::convertToID(std::declval<T>()))>,
            std::is_same<T, decltype(DiscreteActionTraits<T>::convertFromID(std::declval<std::int64_t>()))>>,
        std::nullptr_t> = nullptr>
inline constexpr std::true_type isDiscreteActionHelper(const volatile T*);

inline constexpr std::false_type isDiscreteActionHelper(const volatile void*);

}  // namespace detail

template <class T>
struct IsDiscreteAction
    : public std::conditional_t<
          std::is_reference_v<T>,
          std::false_type,
          decltype(detail::isDiscreteActionHelper(std::declval<T*>()))>
{};

template <class T>
inline constexpr bool IsDiscreteActionV = IsDiscreteAction<T>::value;

template <class T, std::int64_t NumActions>
struct EnumActionTraits
{
	static inline constexpr std::int64_t num_actions = NumActions;
	static_assert(0 < num_actions);
	static std::int64_t convertToID(T action)
	{
		auto id = static_cast<std::underlying_type_t<T>>(action);
		assert(0 <= id && id < num_actions);
		return static_cast<std::int64_t>(id);
	}
	static T convertFromID(std::int64_t id)
	{
		assert(0 <= id && id < num_actions);
		return static_cast<T>(static_cast<std::underlying_type_t<T>>(id));
	}
};

enum class FourDirections : std::uint8_t
{
	UP,
	DOWN,
	LEFT,
	RIGHT,
};

template <>
struct DiscreteActionTraits<FourDirections> : public EnumActionTraits<FourDirections, 4>
{};

enum class FiveDirections : std::uint8_t
{
	NEUTRAL,
	UP,
	DOWN,
	LEFT,
	RIGHT,
};

template <>
struct DiscreteActionTraits<FiveDirections> : public EnumActionTraits<FiveDirections, 5>
{};

enum class EightDirections : std::uint8_t
{
	UP,
	DOWN,
	LEFT,
	RIGHT,
	UP_LEFT,
	UP_RIGHT,
	DOWN_LEFT,
	DOWN_RIGHT,
};

template <>
struct DiscreteActionTraits<EightDirections> : public EnumActionTraits<EightDirections, 8>
{};

enum class NineDirections : std::uint8_t
{
	NEUTRAL,
	UP,
	DOWN,
	LEFT,
	RIGHT,
	UP_LEFT,
	UP_RIGHT,
	DOWN_LEFT,
	DOWN_RIGHT,
};

template <>
struct DiscreteActionTraits<NineDirections> : public EnumActionTraits<NineDirections, 9>
{};

enum class AtariButton : std::uint8_t
{
	NONE,
	FIRE
};

using AtariAction = std::tuple<NineDirections, AtariButton>;

template <>
struct DiscreteActionTraits<AtariAction>
{
	static inline constexpr std::int64_t num_actions = 18;
	static std::int64_t convertToID(AtariAction action)
	{
		auto direction_id = static_cast<std::underlying_type_t<NineDirections>>(std::get<0>(action));
		assert(direction_id < 9);
		auto button_id = static_cast<std::underlying_type_t<AtariButton>>(std::get<1>(action));
		assert(button_id < 2);
		return static_cast<std::int64_t>(direction_id + button_id * 9);
	}
	static AtariAction convertFromID(std::int64_t id)
	{
		assert(0 <= id && id < num_actions);
		auto direction = static_cast<NineDirections>(static_cast<std::underlying_type_t<NineDirections>>(id % 9));
		auto button = static_cast<AtariButton>(static_cast<std::underlying_type_t<AtariButton>>(id / 9));
		return std::make_tuple(direction, button);
	}
};

}  // namespace impala
