# Simplified build for fmha example in lightweight ck-tile version
add_executable(fmha_fwd fmha_fwd.cpp)
target_include_directories(fmha_fwd PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(fmha_fwd PRIVATE -Wno-float-equal)

add_executable(fmha_bwd fmha_bwd.cpp)
target_include_directories(fmha_bwd PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(fmha_bwd PRIVATE -Wno-float-equal)
