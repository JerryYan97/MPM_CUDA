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
		src/main.cpp
		src/shaderProgram/graphicsPipeline.cpp
		src/shaderProgram/graphicsPipeline.h
		src/camera/Camera.cpp
		src/camera/Camera.h
		src/model/model.cpp
		src/model/model.h
		src/model/InstanceModel.cpp
		src/model/InstanceModel.h
		src/simulator/MPMSimulator.cpp
		src/simulator/MPMSimulator.cuh
		src/simulator/MPMCUDA.cu
        src/simulator/material/Elasiticity.cuh
		src/utilities/FilesIO.cpp
		src/utilities/FilesIO.h)

target_include_directories(Jiarui_MPM
		PUBLIC thirdparties/glfw/include
		PUBLIC thirdparties/glad/include
		PUBLIC thirdparties/glm
		PUBLIC thirdparties/imgui
		PUBLIC thirdparties/imgui/backends
		PUBLIC thirdparties/tinyobj
		PUBLIC thirdparties/mesh_query0.1
		PUBLIC thirdparties/cudaSVD
		PUBLIC thirdparties/partio/src/lib)

target_link_libraries(Jiarui_MPM
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/mesh_query0.1/libmesh_query.a
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/glfw/build/src/libglfw3.a
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/partio/build/src/lib/libpartio.so
		${CMAKE_DL_LIBS}
		pthread)

# set_property(TARGET Jiarui_MPM PROPERTY CUDA_ARCHITECTURES 75)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_70,code=sm_70")

if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-DDEBUG=1)
endif()