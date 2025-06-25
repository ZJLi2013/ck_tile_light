# Simplified examples build system for lightweight ck-tile version

# Add all example subdirectories
file(GLOB dir_list LIST_DIRECTORIES true *)

foreach(subdir ${dir_list})
    if(IS_DIRECTORY "${subdir}" AND EXISTS "${subdir}/CMakeLists.txt")
        add_subdirectory(${subdir})
    endif()
endforeach()
