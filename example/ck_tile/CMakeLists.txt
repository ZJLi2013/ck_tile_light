# Add all example subdirectories
file(GLOB example_dirs LIST_DIRECTORIES true *)
foreach(example_dir ${example_dirs})
    if(IS_DIRECTORY "${example_dir}" AND EXISTS "${example_dir}/CMakeLists.txt")
        add_subdirectory(${example_dir})
    endif()
endforeach()
