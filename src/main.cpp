#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <pthread.h>
#include <unistd.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include "shaderProgram/graphicsPipeline.h"
#include "camera/Camera.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

float deltaTime = 0.f;
float lastFrame = 0.f;
int width = 640;
int height = 480;
float lastX = width / 2.0f;
float lastY = height / 2.0f;
bool firstMouse = true;

Camera mCam;

void error_callback(int error, const char* desc){
    fprintf(stderr, "Error: %s\n", desc);
}

void processMovementInput(GLFWwindow *window)
{
    float camSpeed = 5.f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
        mCam.mLookAt += camSpeed * mCam.mViewDir;
        mCam.updateMat();
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
        mCam.mLookAt -= camSpeed * mCam.mViewDir;
        mCam.updateMat();
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
        mCam.mLookAt -= camSpeed * mCam.mRight;
        mCam.updateMat();
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
        mCam.mLookAt += camSpeed * mCam.mRight;
        mCam.updateMat();
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    if(glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS){
        mCam.processMouseMovement(xoffset, yoffset);
    }

}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    mCam.processMouseScroll((float)yoffset);
}

int main() {
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if(!glfwInit()){
        return -1;
    }
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(width, height, "glfwTest", NULL, NULL);
    if(!window){
        std::cout << "GLFW windows creation failed." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

    std::string vPath = std::string(PROJ_PATH) + "/src/shaders/myShader.vert";
    std::string fPath = std::string(PROJ_PATH) + "/src/shaders/myShader.frag";
    glEnable(GL_DEPTH_TEST);
    graphicsPipeline mPipline(vPath,fPath);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig reader_config;
    std::string obj_path = std::string(PROJ_PATH) + "/models/cube.obj";
    if (!reader.ParseFromFile(obj_path)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& shapes = reader.GetShapes();
    auto& attrib = reader.GetAttrib();
    std::vector<float> vert_vec;
    std::vector<unsigned int> ind_vec;
    int vert_num = shapes[0].mesh.num_face_vertices.size() * 3;
    vert_vec.resize(vert_num * 6);
    ind_vec.resize(vert_num);
    for (size_t s = 0; s < shapes.size(); s++) {
        int vert_idx = 0;
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                vert_vec[vert_idx * 6] = vx;
                vert_vec[vert_idx * 6 + 1] = vy;
                vert_vec[vert_idx * 6 + 2] = vz;
                vert_vec[vert_idx * 6 + 3] = nx;
                vert_vec[vert_idx * 6 + 4] = ny;
                vert_vec[vert_idx * 6 + 5] = nz;
                ind_vec[vert_idx] = vert_idx;
                vert_idx++;
            }
            index_offset += fv;
        }
    }

    glm::mat4 model = glm::rotate(glm::mat4(1.f), glm::radians(0.f), glm::vec3(1.f, 0.f, 0.f));
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
    glm::mat3 normalMat(1.0f);
    normalMat[0, 0] = model[0, 0];
    normalMat[0, 1] = model[0, 1];
    normalMat[0, 2] = model[0, 2];
    normalMat[1, 0] = model[1, 0];
    normalMat[1, 1] = model[1, 1];
    normalMat[1, 2] = model[1, 2];
    normalMat[2, 0] = model[2, 0];
    normalMat[2, 1] = model[2, 1];
    normalMat[2, 2] = model[2, 2];
    normalMat = glm::transpose(glm::inverse(normalMat));

    mPipline.setMat3("normalMat", normalMat);
    mPipline.setMat4("model", model);
    mPipline.setMat4("proj", proj);
    std::array<float, 3> lightColor{1.0, 1.0, 1.0};
    mPipline.setVec3("lightColor", lightColor);
    std::array<float, 3> objColor{1.0f, 0.5f, 0.31f};
    mPipline.setVec3("objColor", objColor);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vert_num * 6, vert_vec.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * vert_num, ind_vec.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    float counter = 0.0;
    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();
        processMovementInput(window);

        float curFrame = glfwGetTime();
        deltaTime = curFrame - lastFrame;
        lastFrame = curFrame;

        // Start the Dear ImGui frame

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin(
                    "Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text(
                    "This is some useful text.");               // Display some text (you can use a format strings too)

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float *) &clear_color); // Edit 3 floats representing a color

            if (ImGui::Button(
                    "Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::End();
        }

        counter = counter + 0.01;
        if (counter > 1){
            counter--;
        }
        std::array<float, 4> tmp{counter, 0.0, 0.0, 0.0};
        mPipline.setMat4("view", mCam.mViewMat);
        std::array<float, 3> lightPos{mCam.mPos[0], mCam.mPos[1], mCam.mPos[2]};
        mPipline.setVec3("lightPos", lightPos);
        std::array<float, 3> camPos{mCam.mPos[0], mCam.mPos[1], mCam.mPos[2]};
        mPipline.setVec3("camPos", camPos);

        // Render
        ImGui::Render();
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mPipline.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, vert_num, GL_UNSIGNED_INT, 0);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


        // Swap buffers
        glfwSwapBuffers(window);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    mPipline.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
