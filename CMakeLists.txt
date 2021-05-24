cmake_minimum_required(VERSION 3.17)

cmake_policy(SET CMP0092 NEW) # don't add /W3 for MSVC

project(Jiarui_MPM LANGUAGES CXX CUDA C)

add_definitions( -DPROJ_PATH=\"${CMAKE_CURRENT_LIST_DIR}\")

add_executable(Jiarui_MPM
		thirdparties/glad/src/glad.c
		thirdparties/imgui/imgui.cpp
		thirdparties/imgui/imgui_demo.cpp
		thirdparties/imgui/imgui_draw.cpp
		thirdparties/imgui/imgui_tables.cpp
		thirdparties/imgui/imgui_widgets.cpp
		thirdparties/imgui/backends/imgui_impl_glfw.cpp
		thirdparties/imgui/backends/imgui_impl_glfw.h
		thirdparties/imgui/backends/imgui_impl_opengl3.h
		thirdparties/imgui/backends/imgui_impl_opengl3.cpp
		src/main.cpp src/shaderProgram/graphicsPipeline.cpp src/shaderProgram/graphicsPipeline.h src/camera/Camera.cpp src/camera/Camera.h)

target_include_directories(Jiarui_MPM
		PUBLIC thirdparties/glfw/include
		PUBLIC thirdparties/glad/include
		PUBLIC thirdparties/glm
		PUBLIC thirdparties/imgui
		PUBLIC thirdparties/imgui/backends)

target_link_libraries(Jiarui_MPM
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/glfw/build/src/libglfw3.a
		${CMAKE_DL_LIBS}
		pthread)
