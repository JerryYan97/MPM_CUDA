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
#include "simulator/MPMSimulator.cuh"
#include "utilities/FilesIO.h"
#include <sstream>
#include <iomanip>

float deltaTime = 0.f;
float lastFrame = 0.f;
int width = 640;
int height = 480;
float lastX = width / 2.0f;
float lastY = height / 2.0f;
bool firstMouse = true;
bool process = false;
int outputFrameID = 0;
int frameRate = 36;
float timePerFrame = 1.f / float(frameRate);

Camera mCam(width, height);

void error_callback(int error, const char* desc){
    fprintf(stderr, "Error: %s\n", desc);
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

    float camSpeed = 2.f * deltaTime;
    if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS){
        mCam.mLookAt -= camSpeed * mCam.mRight * xoffset;
        mCam.mLookAt -= camSpeed * mCam.mUp * yoffset;
        mCam.updateMat();
    }
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_M && action == GLFW_PRESS){
        process = true;
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

    // Delete files in Animations
    std::string cmd = "exec rm -r " + std::string(PROJ_PATH) + "/Animations/*";
    system(cmd.c_str());

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
    glfwSwapInterval(0);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

    std::string vPath = std::string(PROJ_PATH) + "/src/shaders/sphereInstance.vert";
    std::string fPath = std::string(PROJ_PATH) + "/src/shaders/sphereInstance.frag";
    glEnable(GL_DEPTH_TEST);
    graphicsPipeline mPipline(vPath,fPath);
    mPipline.cullBackFace();

    std::string vBoundaryPath = std::string(PROJ_PATH) + "/src/shaders/lineDraw.vert";
    std::string fBoundaryPath = std::string(PROJ_PATH) + "/src/shaders/lineDraw.frag";
    graphicsPipeline mBoundaryPipeline(vBoundaryPath, fBoundaryPath);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    std::string obj1_path = std::string(PROJ_PATH) + "/models/cube.obj";
    std::string obj2_path = std::string(PROJ_PATH) + "/models/cylinder.obj";
    std::string obj3_path = std::string(PROJ_PATH) + "/models/sphere.obj";
    std::string obj4_path = std::string(PROJ_PATH) + "/models/MPM2.obj";

    std::vector<ObjInitInfo> mInfoVec;

    // model mModel(obj_path);
    // MPMSimulator mSim(0.1f, 1.0/10000.0, 200, 10, obj1_path, obj2_path);

    // Jello Cube Cylinder case:
    /*
    ObjInitInfo mInfo1;
    mInfo1.objPath = obj1_path;
    mInfo1.initVel = std::array<double, 3>({0.0, -6.0, 0.0});
    mInfo1.initRotationDegree = 35.f;
    mInfo1.initScale = glm::vec3(1.f);
    mInfo1.initTranslation = glm::vec3(5.f, 5.f, 5.f);
    mInfo1.initRotationAxis = glm::normalize(glm::vec3(1.f, 0.f, 0.f));
    mInfo1.mMaterial = Material(1e4, 0.4, 1.1, JELLO);
    mInfoVec.push_back(mInfo1);

    ObjInitInfo mInfo2;
    mInfo2.objPath = obj2_path;
    mInfo2.initVel = std::array<double, 3>({0.0, -1.0, 0.0});
    mInfo2.initRotationDegree = 0.f;
    mInfo2.initScale = glm::vec3(1.f, 2.f, 1.f);
    mInfo2.initTranslation = glm::vec3(5.f, 2.f, 5.f);
    mInfo2.initRotationAxis = glm::normalize(glm::vec3(1.f, 0.f, 0.f));
    mInfo2.mMaterial = Material(1e4, 0.4, 1.1, JELLO);
    mInfoVec.push_back(mInfo2);
    */

    // Jello cube collides boundary wall
    /*
    ObjInitInfo mInfo;
    mInfo.objPath = obj1_path;
    mInfo.initVel = std::array<double, 3>({0.0, -8.0, 0.0});
    mInfo.initRotationDegree = 35.f;
    mInfo.initScale = glm::vec3(1.f);
    mInfo.initTranslation = glm::vec3(5.f, 4.f, 5.f);
    mInfo.initRotationAxis = glm::normalize(glm::vec3(1.f, 0.f, 0.f));
    mInfo.mMaterial = Material(1e4, 0.4, 1.1, JELLO);
    mInfoVec.push_back(mInfo);
    MPMSimulator mSim(0.1f, 1.0/5000.0, 100, 8, mInfoVec);
    */

    // Snow sphere collides wall
    ObjInitInfo mInfo;
    mInfo.objPath = obj4_path;
    mInfo.initVel = std::array<double, 3>({0.0, 0.0, 0.0});
    mInfo.initRotationDegree = 0.f;
    mInfo.initScale = glm::vec3(1.f);
    mInfo.initTranslation = glm::vec3(5.0f, 1.5f, 5.f);
    mInfo.initRotationAxis = glm::normalize(glm::vec3(1.f, 0.f, 0.f));
    mInfo.mMaterial = Material(5e3, 0.3, 0.8, SNOW);
    mInfoVec.push_back(mInfo);

    MPMSimulator mSim(0.025f, 1.0/10000.0, 400, 10, mInfoVec);


    // Snow balls collides
    /*
    ObjInitInfo mInfo1;
    mInfo1.objPath = obj3_path;
    mInfo1.initVel = std::array<double, 3>({-5.0, 0.0, 0.0});
    mInfo1.initRotationDegree = 0.f;
    mInfo1.initScale = glm::vec3(1.f);
    mInfo1.initTranslation = glm::vec3(7.f, 4.f, 5.f);
    mInfo1.initRotationAxis = glm::normalize(glm::vec3(1.f, 0.f, 0.f));
    mInfo1.mMaterial = Material(5e3, 0.3, 400, SNOW);
    mInfoVec.push_back(mInfo1);


    ObjInitInfo mInfo2;
    mInfo2.objPath = obj3_path;
    mInfo2.initVel = std::array<double, 3>({20.0, 0.0, 0.0});
    mInfo2.initRotationDegree = 0.f;
    mInfo2.initScale = glm::vec3(0.5f);
    mInfo2.initTranslation = glm::vec3(3.f, 4.f, 5.f);
    mInfo2.initRotationAxis = glm::normalize(glm::vec3(1.f, 0.f, 0.f));
    mInfo2.mMaterial = Material(5e3, 0.3, 400, SNOW);
    mInfoVec.push_back(mInfo2);

    MPMSimulator mSim(0.05f, 1.0/500.0, 200, 10, mInfoVec);
    */

    // Jello cube collides case:
    /*
    std::vector<double> initVel(mSim.mParticles.particleNum * 3, 0.0);
    for (int i = 0; i < mSim.mParticles.particleNum; ++i) {
        if (i < mSim.mParticles.particleNum / 2){
            initVel[3 * i] = -8.0;
        }else{
            initVel[3 * i] = 8.0;
        }
    }
    mSim.setVel(initVel);
    */

    // Jello cube collides cylinder case:
    /*
    std::vector<double> initVel(mSim.mParticles.particleNum * 3, 0.0);
    for (int i = 0; i < mSim.mParticles.particleNum; ++i) {
        if (i < mSim.mParticles.particleNumDiv[0]){
            initVel[3 * i + 1] = -6.0;
        }else{
            initVel[3 * i + 1] = -1.0;
        }
    }
    mSim.setVel(initVel);
    */




    /*
    std::vector<double> initVel(mSim.mParticles.particleNum * 3, 0.0);
    for (int i = 0; i < mSim.mParticles.particleNum; ++i) {
        initVel[3 * i + 1] = -0.5;
    }
    mSim.setVel(initVel);
    */

    // Init models
    std::string instance_obj_path = std::string(PROJ_PATH) + "/models/sphereLowRes2.obj";
    std::vector<float> insPos;
    mSim.getGLParticlesPos(insPos);
    InstanceModel mModel(instance_obj_path, insPos, 0.01f);

    float upperCorner[3] = {
            static_cast<float>(mSim.mGrid.originCorner[0] + mSim.mGrid.h * mSim.mGrid.nodeNumDim),
            static_cast<float>(mSim.mGrid.originCorner[1] + mSim.mGrid.h * mSim.mGrid.nodeNumDim),
            static_cast<float>(mSim.mGrid.originCorner[2] + mSim.mGrid.h * mSim.mGrid.nodeNumDim)
    };
    model mBoundaryModel(reinterpret_cast<const float *>(mSim.mGrid.originCorner.data()), upperCorner);
    mBoundaryModel.transLinesGRAM();

    mPipline.setMat3("normalMat", mModel.mNormalMat);
    mPipline.setMat4("model", mModel.mModelMat);
    mPipline.setMat4("proj", mCam.mProjMat);
    std::array<float, 3> lightColor{1.0, 1.0, 1.0};
    mPipline.setVec3("lightColor", lightColor);
    std::array<float, 3> objColor{0.1f, 0.1f, 0.9f};
    mPipline.setVec3("objColor", objColor);

    std::array<float, 3> bColor{1.0, 0.0, 0.0};
    mBoundaryPipeline.setVec3("color", bColor);
    mBoundaryPipeline.setMat4("proj", mCam.mProjMat);

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
    while (!glfwWindowShouldClose(window) && mSim.current_time <= 20.0){
        glfwPollEvents();
        // processMovementInput(window);
        process = true;
        if (process){
            mSim.step();
            // process = false;
        }

        // std::vector<float> tmpParticlePos(mSim.mParticles.particlePosVec.begin(), mSim.mParticles.particlePosVec.end());
        std::vector<float> tmpParticlePos;
        mSim.getGLParticlesPos(tmpParticlePos);

        mModel.updateInstanceModel(tmpParticlePos);

        if (int(mSim.current_time / timePerFrame) >= outputFrameID){
            std::stringstream ss;
            ss << std::setw(6) << std::setfill('0') << outputFrameID;
            std::string s = ss.str();
            std::string oFrameNamePath = std::string(PROJ_PATH) + "/Animations/Frame" + s + ".bgeo";
            OutputBego(oFrameNamePath, tmpParticlePos);
            ++outputFrameID;
        }

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

            ImGui::Text("Already Output Frames: %d. Current Sim time: %.2f (s)", outputFrameID, mSim.current_time);
            ImGui::Text("Last time step: %.10f", mSim.adp_dt);

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

        mBoundaryPipeline.setMat4("view", mCam.mViewMat);

        // Render
        ImGui::Render();
        mPipline.renderInstance(mModel);
        mBoundaryPipeline.renderLines(mBoundaryModel);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        // Swap buffers
        glfwSwapBuffers(window);
    }
    mPipline.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
