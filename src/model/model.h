//
// Created by jiaruiyan on 5/26/21.
// Manage a model on RAM or GRAM.

#ifndef JIARUI_MPM_MODEL_H
#define JIARUI_MPM_MODEL_H

#include <vector>
#include <string>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

class model {
protected:


public:
    std::vector<float> mVertData;
    std::vector<unsigned int> mIndices;

    unsigned int VBO{}, EBO{}, VAO{};
    glm::mat4 mModelMat;
    glm::mat3 mNormalMat;
    explicit model(std::string &path);
    void transGRAM();
    int mVertNum;
    ~model();
};


#endif //JIARUI_MPM_MODEL_H
