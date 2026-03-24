# CMakeLists.txt for AMD XDNA NPU Backend

cmake_minimum_required(VERSION 3.14)

project(ggml-amdxdna C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(GGML_AMD_NPU_DEBUG "Enable NPU debug logging" OFF)
option(GGML_AMD_NPU_BUILD_TESTS "Build tests" ON)

# Find dependencies
find_package(PkgConfig QUIET)

# XRT (Xilinx Runtime)
set(XRT_SEARCH_PATHS
    /opt/xilinx/xrt
    /usr/local/xrt
    $ENV{XILINX_XRT}
)

find_path(XRT_INCLUDE_DIR
    NAMES xrt.h
    PATHS ${XRT_SEARCH_PATHS}
    PATH_SUFFIXES include
)

find_library(XRT_CORE_LIBRARY
    NAMES xrt_core xrt_coreutil
    PATHS ${XRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

find_library(XRT_HWEMU_LIBRARY
    NAMES xrt_hwemu
    PATHS ${XRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

if(XRT_INCLUDE_DIR AND XRT_CORE_LIBRARY)
    set(XRT_FOUND TRUE)
    message(STATUS "Found XRT: ${XRT_CORE_LIBRARY}")
else()
    message(STATUS "XRT not found - building stub implementation")
endif()

# Source files
set(GGML_AMDXDNA_SOURCES
    src/ggml-amdxdna.cpp
)

set(GGML_AMDXDNA_HEADERS
    src/ggml-amdxdna.h
)

# Create library
add_library(ggml-amdxdna SHARED
    ${GGML_AMDXDNA_SOURCES}
    ${GGML_AMDXDNA_HEADERS}
)

target_include_directories(ggml-amdxdna
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${XRT_INCLUDE_DIR}
)

# Link dependencies
if(XRT_FOUND)
    target_link_libraries(ggml-amdxdna
        PRIVATE
            ${XRT_CORE_LIBRARY}
            ${XRT_HWEMU_LIBRARY}
            dl
    )
    target_compile_definitions(ggml-amdxdna
        PRIVATE
            GGML_AMDXDNA_XRT_AVAILABLE=1
    )
else()
    target_link_libraries(ggml-amdxdna
        PRIVATE
            dl
    )
    target_compile_definitions(ggml-amdxdna
        PRIVATE
            GGML_AMDXDNA_XRT_AVAILABLE=0
    )
endif()

# Debug logging
if(GGML_AMD_NPU_DEBUG)
    target_compile_definitions(ggml-amdxdna
        PRIVATE
            GGML_AMDXDNA_DEBUG=1
    )
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ggml-amdxdna
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

# Install
install(TARGETS ggml-amdxdna
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${GGML_AMDXDNA_HEADERS}
    DESTINATION include
)

# Python tools
find_package(Python3 QUIET COMPONENTS Interpreter)

if(Python3_FOUND)
    install(PROGRAMS
        tools/gguf_to_onnx.py
        tools/compile_model.py
        tools/benchmark_npu.py
        DESTINATION bin
    )
endif()

# Scripts
install(PROGRAMS
    scripts/detect-npu.sh
    scripts/install-xrt.sh
    scripts/verify_npu.py
    DESTINATION bin
)

# Documentation
install(FILES
    docs/installation.md
    docs/architecture.md
    DESTINATION share/doc/ggml-amdxdna
)

# Tests
if(GGML_AMD_NPU_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_amdxdna
        tests/test_amdxdna.cpp
    )
    
    target_link_libraries(test_amdxdna
        PRIVATE
            ggml-amdxdna
    )
    
    add_test(NAME test_amdxdna
        COMMAND test_amdxdna
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-amdxdnaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ggml-amdxdnaConfig.cmake
    INSTALL_DESTINATION lib/cmake/ggml-amdxdna
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ggml-amdxdnaConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ggml-amdxdnaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ggml-amdxdnaConfigVersion.cmake
    DESTINATION lib/cmake/ggml-amdxdna
)

# Print configuration summary
message(STATUS "")
message(STATUS "ggml-amdxdna Configuration Summary:")
message(STATUS "  XRT found:        ${XRT_FOUND}")
message(STATUS "  XRT include:      ${XRT_INCLUDE_DIR}")
message(STATUS "  XRT library:      ${XRT_CORE_LIBRARY}")
message(STATUS "  Debug logging:    ${GGML_AMD_NPU_DEBUG}")
message(STATUS "  Build tests:      ${GGML_AMD_NPU_BUILD_TESTS}")
message(STATUS "")