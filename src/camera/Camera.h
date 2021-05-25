//
// Created by jiaruiyan on 5/24/21.
//

#ifndef JIARUI_MPM_CAMERA_H
#define JIARUI_MPM_CAMERA_H
#include <glm.hpp>

class Camera {
private:
    void updateCameraVectors();
    float mMouseSensitivity;

public:
    glm::vec3 mPos;
    glm::vec3 mLookAt;
    glm::vec3 mRight;
    glm::vec3 mUp;
    glm::vec3 mViewDir;
    glm::mat4 mViewMat;
    float mPitch;
    float mYaw;
    float mDistant;
    Camera();
    void updateMat();
    void processMouseMovement(float xoffset, float yoffset);
    void processMouseScroll(float yoffset);
};


#endif //JIARUI_MPM_CAMERA_H
