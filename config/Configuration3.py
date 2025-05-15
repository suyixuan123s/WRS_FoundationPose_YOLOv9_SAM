'''# Get the Conda environment prefix dynamically
set(CONDA_ENV_PREFIX $ENV{CONDA_PREFIX})

# Set paths for dependencies (Conda environment paths)
set(CMAKE_PREFIX_PATH
    ${CONDA_ENV_PREFIX}/lib/python3.9/site-packages/pybind11/share/cmake/pybind11
    ${CONDA_ENV_PREFIX}/lib
    ${CONDA_ENV_PREFIX}/include
    ${CONDA_ENV_PREFIX}/include/boost
    ${CONDA_ENV_PREFIX}/include/eigen3
)

# Find packages
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Set compiler flags
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -g3 -O3")

# Include directories (e.g., Eigen and Boost from Conda)
include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Set sources
file(GLOB MY_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Set Pybind11 C++ standard
set(PYBIND11_CPP_STANDARD -std=c++14)

# Create the pybind11 module
pybind11_add_module(mycpp src/app/pybind_api.cpp ${MY_SRC})

# Link libraries
target_link_libraries(mycpp PRIVATE
  ${Boost_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  Eigen3::Eigen
)

'''