cmake_minimum_required(VERSION 3.15)

# Require standard C++17 for all targets.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(lanfear CXX)

# Default to a Release build when nothing is specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Release, RelWithDebInfo, Debug or MinSizeRel)."
      FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Compiled extension is dropped straight into the importable python package.
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lanfear")

set(WARNING_FLAGS
    $<$<CXX_COMPILER_ID:GNU>:
        -Wall -Wextra -Wpedantic
    >
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -Wall -Wextra -Wpedantic -Wno-missing-braces
    >
)

# --- Dependencies -----------------------------------------------------------
# Boost (header-only: math special functions + numeric/odeint). BOOST_ROOT is
# exported by the `boost` module that `load_py313` restores.
find_package(Boost 1.70 REQUIRED)

# OpenMP for the (embarrassingly parallel) coefficient sums and orbit batches.
find_package(OpenMP)

# Locate pybind11 via the interpreter in the active environment, so the .so is
# built for the exact python that will import it.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_status)
if(NOT _pybind11_status EQUAL 0)
  message(FATAL_ERROR "Could not locate pybind11 via ${Python3_EXECUTABLE}. "
                      "Did you run load_py313 first?")
endif()
find_package(pybind11 CONFIG REQUIRED)

# --- Targets ----------------------------------------------------------------
add_library(lanfear_core INTERFACE)
target_include_directories(lanfear_core INTERFACE
  "${PROJECT_SOURCE_DIR}/include")
# Boost as a SYSTEM include so its (noisy) odeint headers don't trip -Wpedantic.
target_include_directories(lanfear_core SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(lanfear_core INTERFACE OpenMP::OpenMP_CXX)
endif()

# Standalone test harness (pure C++, no python) for quick physics checks.
add_executable(test_core "src/test_core.cpp")
target_link_libraries(test_core PRIVATE lanfear_core)
target_compile_options(test_core PRIVATE ${WARNING_FLAGS})

# Python extension module -> lanfear/_core.*.so
pybind11_add_module(_core "src/python_bindings.cpp")
target_link_libraries(_core PRIVATE lanfear_core)
target_compile_options(_core PRIVATE ${WARNING_FLAGS})
