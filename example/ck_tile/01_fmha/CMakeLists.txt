# Build for fmha example with code generation
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api fwd --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api bwd --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/bwd_blob_list.txt --receipt 3
)

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt FMHA_FWD_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/bwd_blob_list.txt FMHA_BWD_GEN_BLOBS)

add_custom_command(
  OUTPUT ${FMHA_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api fwd --output_dir ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
  OUTPUT ${FMHA_BWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api bwd --output_dir ${CMAKE_CURRENT_BINARY_DIR} --receipt 3
)

add_executable(fmha_fwd fmha_fwd.cpp ${FMHA_FWD_GEN_BLOBS})
target_include_directories(fmha_fwd PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(fmha_fwd PRIVATE -Wno-float-equal -Wno-undefined-func-template)

add_executable(fmha_bwd fmha_bwd.cpp ${FMHA_BWD_GEN_BLOBS})
target_include_directories(fmha_bwd PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(fmha_bwd PRIVATE -Wno-float-equal -Wno-undefined-func-template)
