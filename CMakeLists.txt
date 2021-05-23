cmake_minimum_required(VERSION 3.17)

cmake_policy(SET CMP0092 NEW) # don't add /W3 for MSVC

project(Jiarui_MPM LANGUAGES CXX CUDA)

add_executable(Jiarui_MPM
		thirdparties/glad/src/glad.c
		src/main.cpp)

target_include_directories(Jiarui_MPM
		PUBLIC thirdparties/glfw/include
		PUBLIC thirdparties/glad/include)

target_link_libraries(Jiarui_MPM
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/glfw/build/src/libglfw3.a
		${CMAKE_DL_LIBS}
		pthread)
