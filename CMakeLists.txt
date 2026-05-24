cmake_minimum_required(VERSION 3.21)
project(smlx C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Build mlx-c as a subdirectory so we link against the in-tree copy.
add_subdirectory(thirdparty/mlx-c)

add_executable(smlx src/main.c)
target_link_libraries(smlx PRIVATE mlxc)
target_include_directories(smlx PRIVATE thirdparty/mlx-c)
