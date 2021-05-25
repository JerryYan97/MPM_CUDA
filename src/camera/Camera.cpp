//
// Created by jiaruiyan on 5/24/21.
//

#include "Camera.h"
#include <gtc/matrix_transform.hpp>

Camera::Camera() {
    mViewDir = glm::vec3(0.f, 0.f, -1.f);
    mLookAt = glm::vec3 (0.f, 0.f, 0.f);
    mDistant = 1.f;

    mPos = mLookAt + (-mViewDir) * mDistant;
    glm::vec3 revDir = glm::normalize(mPos - mLookAt);
    mUp = glm::vec3(0.f, 1.f, 0.f);
    mRight = glm::normalize(glm::cross(mUp, revDir));
    mViewMat = glm::lookAt(mPos, mLookAt, mUp);
    mYaw = -90.f;
    mPitch = 0.f;
    mMouseSensitivity = 0.2f;
}

void Camera::updateMat() {
    mPos = mLookAt + (-mViewDir) * mDistant;
    mViewMat = glm::lookAt(mPos, mLookAt, mUp);
}

void Camera::processMouseMovement(float xoffset, float yoffset)
{
    xoffset *= mMouseSensitivity;
    yoffset *= mMouseSensitivity;

    mYaw   += xoffset;
    mPitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (mPitch > 89.0f)
        mPitch = 89.0f;
    if (mPitch < -89.0f)
        mPitch = -89.0f;

    // update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    mViewDir.x = cos(glm::radians(mYaw)) * cos(glm::radians(mPitch));
    mViewDir.y = sin(glm::radians(mPitch));
    mViewDir.z = sin(glm::radians(mYaw)) * cos(glm::radians(mPitch));
    mViewDir = glm::normalize(mViewDir);
    mRight = glm::normalize(glm::cross(mUp, -mViewDir));
    mLookAt = mPos + mViewDir * mDistant;
    mViewMat = glm::lookAt(mPos, mLookAt, mUp);
}

void Camera::processMouseScroll(float yoffset) {
    mDistant -= (float)yoffset;
    if (mDistant < 1.0f)
        mDistant = 1.0f;
    if (mDistant > 45.0f)
        mDistant = 45.0f;
    this->updateMat();
}
