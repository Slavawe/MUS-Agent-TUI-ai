# ═══════════════════════════════════════════════════════════════════════
#  МУС (Modular Smart System) — Universal Build System
#  Лицензия: MIT / Apache 2.0
# ═══════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.18)
project(mus_cuda LANGUAGES CXX CUDA
    VERSION 2.0.0
    DESCRIPTION "МУС — Modular Smart System with CUDA"
)

# ─── Global Standards ─────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── Build Options ────────────────────────────────────────────────────
option(MUS_BUILD_CLOUD "Build for cloud (Kaggle/Colab) — no local GPU" OFF)
option(MUS_BUILD_MINIMAL "Build only core + auto trainer" OFF)
option(MUS_ENABLE_DEBUG "Enable debug NaN tracing" OFF)

# ─── Architecture Flags ───────────────────────────────────────────────
# Cross-platform: all NVIDIA generations from Pascal (10xx) to Blackwell (50xx)
#   61  — Pascal    (GTX 1060, 1070, 1080)  ← legacy cloud GPUs
#   75  — Turing    (GTX 1660 Ti, T4, RTX 20xx) ← your GTX 1660 Ti
#   86  — Ampere    (RTX 3060, 3070, 3080, A100)
#   89  — Ada       (RTX 4060, 4070, 4090)
#   90  — Blackwell (RTX 5090, future)
if(MUS_BUILD_CLOUD)
    # Cloud GPUs: T4 (75), P100 (60), A100 (80)
    set(CMAKE_CUDA_ARCHITECTURES "60;75;80")
elseif(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "61;75;86;89;90")
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build mode: $<IF:$<BOOL:${MUS_BUILD_CLOUD}>,Cloud,Local>")

# ─── Compiler Flags ───────────────────────────────────────────────────
if(MSVC)
    # Fix for CUDA 13 + MSVC: standard conforming preprocessor required by CCCL
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/Zc:preprocessor\"")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
    endif()
endif()

if(MUS_ENABLE_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDEBUG_BLOCK_BWD")
    message(STATUS "Debug mode: ON")
endif()

# ─── Optional OpenMP ──────────────────────────────────────────────────
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: found (${OpenMP_CXX_FLAGS})")
else()
    message(STATUS "OpenMP: not found — parallel loops disabled")
endif()

# ─── Dependencies ─────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ═══════════════════════════════════════════════════════════════════════
#  CORE — статическая библиотека со всеми CUDA-кернами
# ═══════════════════════════════════════════════════════════════════════
set(MUS_CORE_SOURCES
    src/kernels.cu
    src/kernels_f16.cu
)

add_library(mus_core STATIC ${MUS_CORE_SOURCES})

target_include_directories(mus_core PUBLIC include)

target_link_libraries(mus_core PUBLIC
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mus_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(mus_core PUBLIC ${OpenMP_CXX_FLAGS})
endif()

set_target_properties(mus_core PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ═══════════════════════════════════════════════════════════════════════
#  Функция добавления инструмента
# ═══════════════════════════════════════════════════════════════════════
function(mus_add_tool NAME SOURCE DESCRIPTION)
    if(MUS_BUILD_MINIMAL AND NOT NAME MATCHES "mus_train_auto|mus_inference|memory_analysis")
        return()
    endif()
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE mus_core)
    set_target_properties(${NAME} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# ═══════════════════════════════════════════════════════════════════════
#  EXECUTABLES
# ═══════════════════════════════════════════════════════════════════════

# ── Core Tools ─────────────────────────────────────────────────────────
mus_add_tool(mus_inference    src/main.cpp          "Inference engine")
mus_add_tool(memory_analysis  src/memory_analysis.cu "VRAM analysis")

# ── Training Tools ─────────────────────────────────────────────────────
mus_add_tool(mus_train        src/train.cu           "FP32 training (121M)")
mus_add_tool(mus_train_f16    src/train_f16.cu       "FP16 training (500M)")
mus_add_tool(mus_train_4_6gb  src/train_4_6gb.cu     "Auto 400M/700M for 4-6GB")

if(NOT MUS_BUILD_MINIMAL)
    mus_add_tool(mus_train_1b_multimodal src/train_1b_multimodal.cu "1B multimodal demo")
    mus_add_tool(mus_train_optimized_8gb src/train_optimized_8gb.cu "Optimized 1B for 8GB")
    mus_add_tool(mus_train_1660ti src/train_1660ti.cu "500M optimized for GTX 1660 Ti")
    mus_add_tool(test_forward_only src/test_forward_only.cu "Forward-only diagnostic")
    mus_add_tool(mus_data_loader         src/data_loader.cu        "Multimodal data loader")
endif()

# ── Test Tools ─────────────────────────────────────────────────────────
mus_add_tool(nan_test        src/nan_test.cu        "Gradient stability test")
mus_add_tool(mus_h100        src/train_h100.cu      "H100 training (test model arch)")

# gemm_test: standalone (no mus_core dependency)
if(NOT MUS_BUILD_MINIMAL)
    add_executable(gemm_test src/gemm_test.cu)
    target_include_directories(gemm_test PRIVATE include)
    target_link_libraries(gemm_test PRIVATE CUDA::cuda_driver CUDA::cudart CUDA::cublas)
endif()

# tokenizer_test: CPU-only C++ tokenizer demo
if(NOT MUS_BUILD_MINIMAL)
    add_executable(tokenizer_test src/tokenizer_test.cpp src/tokenizer.cpp)
    target_include_directories(tokenizer_test PRIVATE include)
    set_target_properties(tokenizer_test PROPERTIES
        LINKER_LANGUAGE CXX
    )
endif()

# ── Cloud-Only Training ───────────────────────────────────────────────
if(MUS_BUILD_CLOUD)
    mus_add_tool(mus_train_kaggle src/train_4_6gb.cu "Cloud auto-trainer")
    target_compile_definitions(mus_train_kaggle PRIVATE MUS_CLOUD_MODE=1)
    message(STATUS "Cloud training target: mus_train_kaggle")
endif()

# ═══════════════════════════════════════════════════════════════════════
#  Install
# ═══════════════════════════════════════════════════════════════════════
install(TARGETS
    mus_inference mus_train mus_train_f16 mus_train_4_6gb
    mus_train_1660ti mus_h100
    memory_analysis gemm_test nan_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ═══════════════════════════════════════════════════════════════════════
#  Post-build summary
# ═══════════════════════════════════════════════════════════════════════
message(STATUS "══════════════════════════════════════")
message(STATUS "  МУС Build Summary:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Standards:    C++17 / CUDA17")
message(STATUS "  Arches:       ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Cloud mode:   ${MUS_BUILD_CLOUD}")
message(STATUS "  Minimal:      ${MUS_BUILD_MINIMAL}")
message(STATUS "══════════════════════════════════════")