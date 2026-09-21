cmake_minimum_required(VERSION 3.20)
project(supervram LANGUAGES CXX)

option(SUPERVRAM_BUILD_TESTS "Build native tests" ON)

add_library(supervram_native STATIC src/supervram_native.cpp)
target_include_directories(supervram_native PUBLIC include)
target_compile_features(supervram_native PUBLIC cxx_std_17)
target_link_libraries(supervram_native PUBLIC ${CMAKE_DL_LIBS})

if(SUPERVRAM_BUILD_TESTS)
    enable_testing()
    add_executable(test_supervram_native tests/test_native.cpp)
    target_link_libraries(test_supervram_native PRIVATE supervram_native)
    add_test(NAME supervram_native COMMAND test_supervram_native)
endif()
