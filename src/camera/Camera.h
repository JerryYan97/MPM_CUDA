//
// Created by jiaruiyan on 5/24/21.
//

#ifndef JIARUI_MPM_CAMERA_H
#define JIARUI_MPM_CAMERA_H
#include <glm.hpp>

class Camera {
private:
    glm::vec3 mPos;
    glm::vec3 mLookAt;

public:
    Camera();
};


#endif //JIARUI_MPM_CAMERA_H
