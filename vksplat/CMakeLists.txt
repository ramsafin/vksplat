cmake_minimum_required(VERSION 3.16)
project(VkSplat)

if (MSVC)
    set(CMAKE_CXX_FLAGS "/W4")
    # set(CMAKE_CXX_FLAGS_DEBUG "/Zi /fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    # set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer -fno-common")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Find required packages
find_package(Vulkan REQUIRED)

if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    set(VKSPLAT_BUILD_PYTHON_DEFAULT OFF)
else()
    set(VKSPLAT_BUILD_PYTHON_DEFAULT ON)
endif()
option(VKSPLAT_BUILD_PYTHON "Build the pybind11 Python extension" ${VKSPLAT_BUILD_PYTHON_DEFAULT})

if(VKSPLAT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    # Find pybind11
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        # If pybind11 is not found, download it
        message(STATUS "Downloading pybind11")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG        v2.11.1
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Find GLM
find_package(glm QUIET)
if(NOT glm_FOUND)
    # If GLM is not found, download it
    message(STATUS "Downloading GLM")
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG        0.9.9.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(glm)
endif()

# Core source files shared by Python bindings and native executables.
set(VKSPLAT_CORE_SOURCES
    src/buffer.cpp
    src/colmap_reader.cpp
    src/knn.cpp
    src/scheduler.cpp
    src/gs_pipeline.cpp
    src/gs_renderer.cpp
    src/gs_trainer.cpp
    src/perf_timer.cpp
    src/training_session.cpp
)

set(VKSPLAT_CORE_HEADERS
    src/config.h
    src/buffer.h
    src/colmap_reader.h
    src/knn.h
    src/scheduler.h
    src/gs_pipeline.h
    src/gs_renderer.h
    src/gs_trainer.h
    src/perf_timer.h
    src/training_session.h
)

add_library(vksplat_core
    ${VKSPLAT_CORE_SOURCES}
    ${VKSPLAT_CORE_HEADERS}
)

target_include_directories(vksplat_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(vksplat_core PUBLIC cxx_std_17)
set_target_properties(vksplat_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(vksplat_core PUBLIC
    Vulkan::Vulkan
    glm::glm
    # asan
)

if(DEFINED OPTIMIZATION_LEVEL)
    target_compile_definitions(vksplat_core PUBLIC
        "OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL}")
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(vksplat_core PUBLIC VK_USE_PLATFORM_WIN32_KHR)
elseif(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    # VkSplat currently uses compute-only Vulkan and does not create surfaces on
    # OHOS.  Do not define Xlib/Win32/MoltenVK platform macros for this target.
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(vksplat_core PUBLIC VK_USE_PLATFORM_XLIB_KHR)
elseif(APPLE)
    target_compile_definitions(vksplat_core PUBLIC VK_USE_PLATFORM_MACOS_MVK)
endif()

if(VKSPLAT_BUILD_PYTHON)
    # Create the Python module from the bindings adapter only.
    pybind11_add_module(vksplat src/python_bindings.cpp)
    target_link_libraries(vksplat PRIVATE vksplat_core)
    target_compile_definitions(vksplat PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

    # Set properties
    set_target_properties(vksplat PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endif()

# Native C++ training executable. It uses the reusable core library directly and
# does not link against Python, pybind11, NumPy, or the Python viewer.
add_executable(vksplat_train apps/train_app.cpp)
target_link_libraries(vksplat_train PRIVATE vksplat_core)
target_compile_features(vksplat_train PRIVATE cxx_std_17)

# Install target
if(VKSPLAT_BUILD_PYTHON)
    install(TARGETS vksplat DESTINATION .)
endif()
install(TARGETS vksplat_train DESTINATION .)
