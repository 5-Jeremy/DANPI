cmake_minimum_required(VERSION 3.16)
project(DANPI) # Fill in version number and description later

include_directories (../AudioFile)
add_executable (DANPI main.cpp)
target_compile_features(DANPI PRIVATE cxx_std_17)