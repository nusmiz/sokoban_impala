#pragma once

#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

#include "action.hpp"

namespace impala
{

enum class EnvState : std::uint8_t
{
	RUNNING,
	FINISHED
};

namespace detail
{

template <class T,
    std::enable_if_t<
        std::conjunction_v<
            IsDiscreteAction<typename T::Action>,
            std::is_invocable_r<typename T::Observation, decltype(std::mem_fn(&T::reset)), T*>,
            std::is_invocable_r<std::tuple<typename T::Observation, typename T::Reward, EnvState>, decltype(std::mem_fn(&T::step)), T*, typename T::Action>,
            std::is_invocable_r<void, decltype(std::mem_fn(&T::render)), const T*>,
            std::is_same<typename T::Reward, decltype(T::decayReward(std::declval<typename T::Reward>()))>,
            std::is_same<typename T::ObsBatch, decltype(T::makeBatch(std::declval<std::vector<typename T::Observation>&>().begin(), std::declval<std::vector<typename T::Observation>&>().end()))>,
            std::is_same<typename T::ObsBatch, decltype(T::makeBatch(std::declval<std::vector<std::reference_wrapper<std::add_const_t<typename T::Observation>>>&>().begin(), std::declval<std::vector<std::reference_wrapper<std::add_const_t<typename T::Observation>>>&>().end()))>,
            std::is_same<typename T::NetworkInput, decltype(T::batchToNetworkInput(std::declval<std::add_lvalue_reference_t<typename T::ObsBatch>>()))>>,
        std::nullptr_t> = nullptr>
inline constexpr std::true_type isEnvironmentHelper(const volatile T*);

inline constexpr std::false_type isEnvironmentHelper(const volatile void*);

}  // namespace detail

template <class T>
struct IsEnvironment
    : public std::conditional_t<
          std::is_reference_v<T> || std::is_const_v<T> || std::is_volatile_v<T>,
          std::false_type,
          decltype(detail::isEnvironmentHelper(std::declval<T*>()))>
{};

template <class T>
inline constexpr bool IsEnvironmentV = IsEnvironment<T>::value;


}  // namespace impala
