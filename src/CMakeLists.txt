cmake_minimum_required(VERSION 3.18)
project(DANPI VERSION "1.0.0" LANGUAGES CXX CUDA)
include_directories (../AudioFile)
add_executable (DANPI main.cu)
set_target_properties(DANPI PROPERTIES CUDA_ARCHITECTURES "75")
target_compile_features(DANPI PRIVATE cxx_std_17)
target_compile_features(DANPI PRIVATE cuda_std_11)