set(PROJECT "XLIB")
project(${PROJECT})

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA REQUIRED)

message("\nTest on cmake v" ${CMAKE_VERSION} "  System: " ${CMAKE_SYSTEM} "\n")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

if(CUDA_VERSION_MAJOR LESS 7)
	message(FATAL_ERROR "\nRequired Cuda Toolkit >= v7.0   Founded: Cuda Toolkit v${CUDA_VERSION_MAJOR}")
endif()

# ------------------------------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR}/include/
                    ${PROJECT_SOURCE_DIR}/Graph/include/)

set(SRC 	    ${PROJECT_SOURCE_DIR}/src/fUtil.cpp
				${PROJECT_SOURCE_DIR}/src/Numeric.cpp
                ${PROJECT_SOURCE_DIR}/src/Timer.cpp
                ${PROJECT_SOURCE_DIR}/src/Cuda/Timer.cu
                ${PROJECT_SOURCE_DIR}/src/Cuda/cuda_util.cu )

# ------------------------------------------------------------------------------

set(HOST_W_FLAGS "-Wall -Wextra -Wno-deprecated-declarations -Wno-missing-field-initializers -fno-diagnostics-show-caret -ftrack-macro-expansion=0")
set(DEVICE_W_FLAGS -Wall,-Wextra,-Wno-deprecated-declarations,-Wno-missing-field-initializers,-fno-diagnostics-show-caret,-ftrack-macro-expansion=0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOST_W_FLAGS}")


set(CUDA_PROPAGATE_HOST_FLAGS OFF)

#-keep -keep-dir ./TMP
set(CUDA_NVCC_FLAGS -std=c++11 -Xcompiler=-O3,${DEVICE_W_FLAGS})


set(ARCH "" CACHE STRING "architectures")

#release options
set(CUDA_NVCC_FLAGS_RELEASE "-use_fast_math" "--expt-relaxed-constexpr" "--resource-usage" "-Xcompiler=-O3,-march=native,-funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
# ------------------------------------------------------------------------------

CUDA_ADD_LIBRARY( XLib_sm30 ${SRC} STATIC OPTIONS "-arch=sm_30" "-DARCH=300")
CUDA_ADD_LIBRARY( XLib_sm35 ${SRC} STATIC OPTIONS "-arch=sm_35" "-DARCH=350")
CUDA_ADD_LIBRARY( XLib_sm52 ${SRC} STATIC OPTIONS "-arch=sm_52" "-DARCH=520")

add_custom_target(clean-all rm -rf *)

set(EXECUTABLE_OUTPUT_PATH .)
