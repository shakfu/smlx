cmake_minimum_required(VERSION 3.21)
project(smlx C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Build mlx-c as a subdirectory so we link against the in-tree copy.
add_subdirectory(thirdparty/mlx-c)

# libsmlx: model loading, forward, sampling, KV cache. Public API in src/smlx.h.
add_library(smlx_core STATIC src/smlx.c)
target_include_directories(smlx_core PUBLIC src thirdparty/mlx-c)
target_link_libraries(smlx_core PUBLIC mlxc)
set_target_properties(smlx_core PROPERTIES OUTPUT_NAME smlx)

# CLI binary.
add_executable(smlx src/main.c)
target_link_libraries(smlx PRIVATE smlx_core)

# Tiny example showing how to link against libsmlx without the CLI.
add_executable(hello_lib examples/hello_lib.c)
target_link_libraries(hello_lib PRIVATE smlx_core)
