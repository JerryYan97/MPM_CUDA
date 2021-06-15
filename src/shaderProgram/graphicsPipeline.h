//
// Created by jiaruiyan on 5/24/21.
//

#ifndef JIARUI_MPM_GRAPHICSPIPELINE_H
#define JIARUI_MPM_GRAPHICSPIPELINE_H

#include <iostream>
#include <array>
#include <glm.hpp>
#include "../model/InstanceModel.h"

class graphicsPipeline {
private:
    unsigned int ID;
public:
    graphicsPipeline(std::string& vPath, std::string& fPath);
    void use();
    void destroy();

    void setVec3(const std::string& name, std::array<float, 3>& val);
    void setVec4(const std::string& name, std::array<float, 4>& val);

    void setMat3(const std::string& name, glm::mat3& val);
    void setMat4(const std::string& name, glm::mat4& val);

    void render(model& renderedModel);
    void renderInstance(InstanceModel& renderedModel);
    void renderLines(model &renderedModel);

    void cullBackFace();
};


#endif //JIARUI_MPM_GRAPHICSPIPELINE_H
