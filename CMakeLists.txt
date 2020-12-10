cmake_minimum_required(VERSION 3.17)

cmake_policy(SET CMP0092 NEW) # don't add /W3 for MSVC

project(Jiarui_MPM LANGUAGES CXX CUDA)

add_executable(Jiarui_MPM)

target_compile_features(Jiarui_MPM PUBLIC cxx_std_20)

target_sources(Jiarui_MPM
	PRIVATE
		"src/main.cpp"
)

find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

target_link_libraries(Jiarui_MPM glfw imgui::imgui)