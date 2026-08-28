cmake_minimum_required(VERSION 3.15)
project(parakeet_stt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/backend.cpp)

# ---------------------------------------------------------------------------
# TRACK A SEAM
#
# With no flags this builds the stub backend: it compiles, imports, and returns
# canned text, so the whole Python package and its packaging are exercised for
# real from the first minute.
#
# To wire in the real engine, configure with the parakeet.cpp tree and the
# built library:
#
#   pip install . --config-settings=cmake.define.PARAKEET_ROOT=/opt/parakeet.cpp \
#                 --config-settings=cmake.define.PARAKEET_LIB=/opt/parakeet.cpp/build-shared/libparakeet.so
#
# Nothing else in the project changes. src/backend.cpp switches on
# PARAKEET_NATIVE and calls the real C-API instead of the stub.
# ---------------------------------------------------------------------------
set(PARAKEET_ROOT "" CACHE PATH "Path to the parakeet.cpp source tree")
set(PARAKEET_LIB  "" CACHE FILEPATH "Path to the built parakeet shared library")

if(PARAKEET_ROOT AND PARAKEET_LIB)
  message(STATUS "parakeet-stt: NATIVE backend (${PARAKEET_LIB})")
  target_compile_definitions(_core PRIVATE PARAKEET_NATIVE=1)
  target_include_directories(_core PRIVATE "${PARAKEET_ROOT}/include")
  target_link_libraries(_core PRIVATE "${PARAKEET_LIB}")
else()
  message(STATUS "parakeet-stt: STUB backend (set PARAKEET_ROOT and PARAKEET_LIB for native)")
endif()

install(TARGETS _core DESTINATION parakeet_stt)
