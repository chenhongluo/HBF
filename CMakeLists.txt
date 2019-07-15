set(PROJECT "HBF")
project(${PROJECT})

cmake_minimum_required(VERSION 3.0)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost)

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

set(XLIB_DIR    ${PROJECT_SOURCE_DIR}/XLib)

include_directories(${PROJECT_SOURCE_DIR}/include/
                    ${PROJECT_SOURCE_DIR}/cub-1.4.1/
                    ${XLIB_DIR}/
                    ${XLIB_DIR}/Graph/
)

set(HOST_SRC 	    ${PROJECT_SOURCE_DIR}/src/Host/GraphSSSP.cpp
                    ${PROJECT_SOURCE_DIR}/src/Host/BellmanFordQueue.cpp
                    ${PROJECT_SOURCE_DIR}/src/Host/DijkstraSET.cpp
)

if (Boost_FOUND)
    set(HOST_SRC ${HOST_SRC}
                 ${PROJECT_SOURCE_DIR}/src/Host/BoostDijkstra.cpp
                 ${PROJECT_SOURCE_DIR}/src/Host/BoostBellmanFord.cpp)
    set(BOOST_FLAG -DBOOST_FOUND)
endif()

set(DEVICE_SRC 	    ${PROJECT_SOURCE_DIR}/src/Device/HBFGraph.cu
                    ${PROJECT_SOURCE_DIR}/src/Device/DegreeMarker.cu
                    ${PROJECT_SOURCE_DIR}/src/Device/WorkEfficient.cu
)

set(HOST_UTIL_SRC 	${XLIB_DIR}/Base/Host/src/fUtil.cpp
                    ${XLIB_DIR}/Base/Host/src/Timer.cpp
                    ${XLIB_DIR}/Graph/Host/src/readHeader.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphBase.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphDegree.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphDegree_Read.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphSTD.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphSTD_Read.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphSTD_BFS.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphWeight.cpp
                    ${XLIB_DIR}/Graph/Host/src/GraphWeight_Read.cpp
)

set(DEVICE_UTIL_SRC ${XLIB_DIR}/Base/Device/Util/src/cuda_util.cu
                    ${XLIB_DIR}/Base/Device/Util/src/global_sync.cu
                    ${XLIB_DIR}/Base/Device/Util/src/Timer.cu
                    ${XLIB_DIR}/Graph/Device/src/cudaGraph.cu
                    ${XLIB_DIR}/Graph/Device/src/cudaGraphWeight.cu
)

#set(HOST_W_FLAGS "-Wall -Wextra")
#set(DEVICE_W_FLAGS -Wall,-Wextra)

#set(HOST_W_FLAGS "-Wall -Wextra -Wno-deprecated-declarations -Wno-missing-field-initializers -fno-diagnostics-show-caret -ftrack-macro-expansion=0")
#set(DEVICE_W_FLAGS -Wall,-Wextra,-Wno-deprecated-declarations,-Wno-missing-field-initializers,-fno-diagnostics-show-caret,-ftrack-macro-expansion=0)

#if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
#    SET(HOST_W_FLAGS "${HOST_W_FLAGS} -fno-diagnostics-show-caret -ftrack-macro-expansion=0")
#    SET(DEVICE_W_FLAGS "${DEVICE_W_FLAGS},-fno-diagnostics-show-caret,-ftrack-macro-expansion=0")
#endif()

# ------------------------------------------------------------------------------

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOST_W_FLAGS}")
# ------------------------------------------------------------------------------

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_NVCC_FLAGS "-std=c++11" "-Xcompiler=-O3" "-w")

set(SM "" CACHE STRING "number_of_SM")
set(ARCH "" CACHE STRING "architectures")

if (ARCH STREQUAL "30")
	message("\nGPU architecture: 3.0\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30" "-DARCH=300")
elseif (ARCH STREQUAL "32")
    message("\nGPU architecture: 3.2\n")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_32" "-DARCH=320")
elseif (ARCH STREQUAL "35")
	message("\nGPU architecture: 3.5\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35" "-DARCH=350" "-rdc=true")
elseif (ARCH STREQUAL "35")
    message("\nGPU architecture: 3.7\n")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_37" "-DARCH=370")
elseif (ARCH STREQUAL "50")
	message("\nGPU architecture: 5.0\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50" "-DARCH=500")
elseif (ARCH STREQUAL "52")
	message("\nGPU architecture: 5.2\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52" "-DARCH=520")
elseif (ARCH STREQUAL "53")
	message("\nGPU architecture: 5.3\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_53" "-DARCH=530")
else()
	message(FATAL_ERROR "\nplease select GPU architecture: \"cmake -DARCH=[ 30 | 32 | 35 | 37 | 50 | 52 | 53 ]\"")
endif()

# ------------------------------------------------------------------------------
if (CMAKE_BUILD_TYPE STREQUAL "")
	set (CMAKE_BUILD_TYPE Release)
endif()

message("Selected CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} "\n")
if (    (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
	AND (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
)
	message(FATAL_ERROR "\nwrong BUILD_TYPE: \"-D CMAKE_BUILD_TYPE=[ Release | Debug ]\"")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#debug options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g" "-G")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
#release options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-use_fast_math" "-Xcompiler=-O3,-march=native,-funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops")
endif()
# ------------------------------------------------------------------------------

find_library(
	DEVRT_LIBRARY cudadevrt
    HINTS  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

cuda_add_executable(HBF ${PROJECT_SOURCE_DIR}/Main.cu ${HOST_SRC} ${DEVICE_SRC}
                        ${HOST_UTIL_SRC} ${DEVICE_UTIL_SRC}
                        OPTIONS -DSM=${SM} ${BOOST_FLAG})

set_target_properties(HBF PROPERTIES COMPILE_FLAGS ${BOOST_FLAG})

target_link_libraries(HBF ${DEVRT_LIBRARY})

add_custom_target(clean-all rm -rf *)

set(EXECUTABLE_OUTPUT_PATH .)

# SET CC and CXX
