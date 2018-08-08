cmake_minimum_required(VERSION 3.5)

project(impala CXX)

FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs 3 REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python3 numpy3 system regex REQUIRED)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -pedantic -Wconversion -Wcast-qual")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

set(impala_source
    main.cpp
    sokoban_env.cpp
    network.cpp)

add_executable(impala ${impala_source})
target_include_directories(impala PRIVATE .)
target_include_directories(impala SYSTEM PRIVATE ./range-v3/include)
target_include_directories(impala PRIVATE ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(impala PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Threads::Threads stdc++fs)
