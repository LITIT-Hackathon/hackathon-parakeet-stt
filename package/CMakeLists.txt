cmake_minimum_required(VERSION 3.20)
project(parakeet_stt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Native engine
#
# By default this fetches parakeet.cpp at a pinned commit and compiles it
# straight into the _core extension, so `pip install .` produces a working
# native library with no extra scripts, no separate shared object, and no
# LD_LIBRARY_PATH / rpath juggling. A from-source install needs a C++17
# toolchain, CMake, and (once) network access for the fetch.
#
#   -DPARAKEET_STT_BUNDLED=OFF     build the canned-text stub instead
#                                  (offline, CI-fast, no compiler for the engine)
#   -DPARAKEET_STT_GIT_TAG=<sha>   vendor a different parakeet.cpp commit
#   -DPARAKEET_STT_MARCH_NATIVE=OFF portable build (for redistributable wheels;
#                                  the default bakes in -march=native)
#   -DFETCHCONTENT_SOURCE_DIR_PARAKEET_CPP=<path>
#                                  use a local parakeet.cpp checkout, skip the clone
# ---------------------------------------------------------------------------
option(PARAKEET_STT_BUNDLED "Fetch and build parakeet.cpp into the extension" ON)
option(PARAKEET_STT_MARCH_NATIVE "Let ggml optimise for the build CPU (-march=native)" ON)
set(PARAKEET_STT_GIT_TAG "e75de9b6b9b688fd293aa22f7e27aa724ea286f8"
    CACHE STRING "parakeet.cpp commit to vendor")

if(PARAKEET_STT_BUNDLED)
  include(FetchContent)

  # parakeet.cpp builds a CLI and an OpenAI-style server by default; we only
  # want the library. Force these before MakeAvailable so the cache picks them up.
  set(PARAKEET_BUILD_CLI    OFF CACHE BOOL "" FORCE)
  set(PARAKEET_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(PARAKEET_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
  set(PARAKEET_SHARED       OFF CACHE BOOL "" FORCE)  # static, linked into _core

  # ggml would otherwise build libggml.so.* and _core would need it on the
  # loader path at import time. Static everything into the one extension module.
  set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS      OFF CACHE BOOL "" FORCE)

  # ggml forces GGML_NATIVE ON (-march=native) unless told otherwise. Fine for a
  # build-where-you-run install; a wheel meant to ship must set this OFF or it
  # will SIGILL on older CPUs.
  if(NOT PARAKEET_STT_MARCH_NATIVE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  endif()

  FetchContent_Declare(
    parakeet_cpp
    GIT_REPOSITORY https://github.com/mudler/parakeet.cpp.git
    GIT_TAG        ${PARAKEET_STT_GIT_TAG}
    GIT_SUBMODULES_RECURSE ON      # pulls third_party/ggml
    GIT_PROGRESS   ON
    # parakeet.cpp applies in-tree ggml patches (broadcast-fold, etc.) from its
    # own top-level CMakeLists via ${CMAKE_SOURCE_DIR}/scripts/... — which points
    # at THIS project when it is a subproject, so that step silently no-ops.
    # Run it here, before ggml is configured. The script is idempotent.
    PATCH_COMMAND bash scripts/apply_ggml_patches.sh
  )
  FetchContent_MakeAvailable(parakeet_cpp)

  # Same CMAKE_SOURCE_DIR problem: parakeet.cpp's library target looks for its
  # vendored dr_wav.h under ${CMAKE_SOURCE_DIR}/third_party. Point it at the real
  # checkout so audio_io.cpp compiles.
  target_include_directories(parakeet PRIVATE ${parakeet_cpp_SOURCE_DIR}/third_party)
endif()

pybind11_add_module(_core src/backend.cpp)

if(PARAKEET_STT_BUNDLED)
  # parakeet.cpp's `parakeet` target carries its PUBLIC include/ (parakeet_capi.h)
  # and links ggml transitively, so linking it is all _core needs.
  target_compile_definitions(_core PRIVATE PARAKEET_NATIVE=1)
  target_link_libraries(_core PRIVATE parakeet)
  message(STATUS "parakeet-stt: NATIVE backend (parakeet.cpp @ ${PARAKEET_STT_GIT_TAG})")
else()
  message(STATUS "parakeet-stt: STUB backend (PARAKEET_STT_BUNDLED=OFF)")
endif()

install(TARGETS _core DESTINATION parakeet_stt)
