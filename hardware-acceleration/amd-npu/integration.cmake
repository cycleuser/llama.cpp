# AMD NPU Integration for llama.cpp
# Add this to the main CMakeLists.txt

# Option to enable AMD NPU support
option(GGML_AMD_NPU "Enable AMD XDNA NPU backend" OFF)

if(GGML_AMD_NPU)
    message(STATUS "AMD NPU backend enabled")
    
    set(AMD_NPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hardware-acceleration/amd-npu")
    
    if(EXISTS "${AMD_NPU_DIR}/CMakeLists.txt")
        add_subdirectory("${AMD_NPU_DIR}" "${CMAKE_BINARY_DIR}/ggml-amd-npu")
        
        # Link AMD NPU backend to main library
        target_link_libraries(ggml PRIVATE ggml-amdxdna)
        target_include_directories(ggml PRIVATE "${AMD_NPU_DIR}/src")
        
        target_compile_definitions(ggml PRIVATE GGML_USE_AMDXDNA)
        
        message(STATUS "AMD NPU backend integrated")
    else()
        message(WARNING "AMD NPU backend directory not found")
    endif()
endif()

# Backend registration
if(GGML_AMD_NPU)
    set(GGML_SOURCES_AMD_NPU
        "${AMD_NPU_DIR}/src/ggml-amdxdna.cpp"
    )
    
    list(APPEND GGML_SOURCES ${GGML_SOURCES_AMD_NPU})
endif()