cmake_minimum_required(VERSION 3.1)

project(main LANGUAGES CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}") 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mtune=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd
set(CMAKE_COLOR_DIAGNOSTICS ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CXX_STANDARD 17)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    cmake_policy(SET CMP0104 OLD) # for CUDA_ARCHITECTURES
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -std=c++17")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
else()
    set(CMAKE_CXX_STANDARD 20) # Use C++20 for CPU compilation to use the new bit field operations. Enabling C++20 for CUDA compilation fails.
    message(STATUS "CUDA not found. The program will be compiled for CPU.")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-xc++>) # force compilation as .cpp files for .cu files
    find_package(OpenMP)
    if(OpenMP_FOUND)
        message(STATUS "OpenMP found.")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DVIENNACL_WITH_OPENMP")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DVIENNACL_WITH_OPENMP")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}    ${OpenMP_EXE_LINKER_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_MODULE_LINKER_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
        set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP")
    else()
        message(STATUS "OpenMP not found.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
    endif()

    # If CUDA is not found, use the thrust library from the submodule instead.
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/thrust)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/thrust/dependencies/cub)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/thrust/dependencies/libcudacxx/include)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/lbvh/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/json/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file( GLOB APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cu )
foreach( SRCFILES ${APP_SOURCES} )
    file(RELATIVE_PATH EXECUTABLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} ${SRCFILES})

    #Remove ".cu" and "apps/" to get the name of the executable.
    string( REPLACE ".cu" "" EXECUTABLE_NAME ${EXECUTABLE_NAME} )
    string( REPLACE "apps/" "" EXECUTABLE_NAME ${EXECUTABLE_NAME} )
    if(NOT CMAKE_CUDA_COMPILER)
    set_source_files_properties(${SRCFILES} PROPERTIES LANGUAGE CXX)
    endif()
    add_executable( ${EXECUTABLE_NAME} ${SRCFILES} )
endforeach()