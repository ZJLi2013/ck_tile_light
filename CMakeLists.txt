cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0140)
  # policies CMP0140 not known to CMake until 3.25
  cmake_policy(SET CMP0140 NEW)
endif()

get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# Simplified build system for lightweight ck-tile version
cmake_minimum_required(VERSION 3.14)

project(ck_tile_light VERSION 1.0.0 LANGUAGES CXX HIP)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

# Default installation path
if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Installation directory")
endif()

# Find Python (required for some code generation)
find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)

# Set default data types
add_definitions(-DCK_ENABLE_INT8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32 -DCK_ENABLE_FP64 -DCK_ENABLE_BF16 -DCK_ENABLE_FP8 -DCK_ENABLE_BF8)

# Find HIP
find_package(hip REQUIRED)
add_compile_definitions(__HIP_PLATFORM_AMD__=1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create config file
configure_file(include/ck_tile/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ck_tile/config.h)

# Create interface library for header-only ck-tile
add_library(ck_tile INTERFACE)
target_include_directories(ck_tile INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${HIP_INCLUDE_DIRS}
    # Add uk directories for flatmm includes
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ck_tile/ops/flatmm/block/uk
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ck_tile/ops/flatmm/block    
)

# Include standard installation directories
include(GNUInstallDirs)

# Build examples if requested
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

# Create and install package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ck_tileConfigVersion.cmake"
    VERSION "1.0.0"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ck_tileConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ck_tile
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ck_tile
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

# Install config.h
install(FILES ${PROJECT_BINARY_DIR}/include/ck_tile/config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck_tile)

# Install package config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ck_tileConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ck_tileConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ck_tile
)
