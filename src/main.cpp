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
#include "model/model.h"
#include "mesh_query.h"
#include "simulator/MPMSimulator.h"

float deltaTime = 0.f;
float lastFrame = 0.f;
int width = 640;
int height = 480;
float lastX = width / 2.0f;
float lastY = height / 2.0f;
bool firstMouse = true;

Camera mCam(width, height);

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

/* */
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
    glfwSwapInterval(1);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

    // std::string vPath = std::string(PROJ_PATH) + "/src/shaders/myShader.vert";
    // std::string fPath = std::string(PROJ_PATH) + "/src/shaders/myShader.frag";
    std::string vPath = std::string(PROJ_PATH) + "/src/shaders/sphereInstance.vert";
    std::string fPath = std::string(PROJ_PATH) + "/src/shaders/sphereInstance.frag";
    glEnable(GL_DEPTH_TEST);
    graphicsPipeline mPipline(vPath,fPath);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    std::string obj_path = std::string(PROJ_PATH) + "/models/sphere.obj";
    // model mModel(obj_path);
    MPMSimulator mSim;
    std::vector<float> insPos{
        5.f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        -5.f, 0.f, 0.f
    };


    InstanceModel mModel(obj_path, insPos);

    mPipline.setMat3("normalMat", mModel.mNormalMat);
    mPipline.setMat4("model", mModel.mModelMat);
    mPipline.setMat4("proj", mCam.mProjMat);
    std::array<float, 3> lightColor{1.0, 1.0, 1.0};
    mPipline.setVec3("lightColor", lightColor);
    std::array<float, 3> objColor{0.1f, 0.1f, 0.9f};
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

        insPos[0] += 0.01;
        insPos[3] += 0.01;
        insPos[6] += 0.01;

        mModel.updateInstanceModel(insPos);
        // Render
        ImGui::Render();
        mPipline.renderInstance(mModel);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        // Swap buffers
        glfwSwapBuffers(window);
    }
    mPipline.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
