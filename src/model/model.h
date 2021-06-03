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
    bool mUseInGL;
public:
    unsigned int VBO{}, EBO{}, VAO{};
    glm::mat4 mModelMat;
    glm::mat3 mNormalMat;

    std::vector<float> mGLVertData;
    std::vector<unsigned int> mGLIndices;
    std::vector<int> mQMIndData; // Query mesh indices: For MPM sampling.
    std::vector<double> mQMVertData;
    std::vector<unsigned int> mVertLength; // Support multiple models.

    glm::vec3 mUpperBound;
    glm::vec3 mLowerBound;

    explicit model(std::string &path, float modelSize, bool usedGL);
    void transGRAM();
    void setTransformation(glm::vec3 scale, glm::vec3 translation, float rotateDegree, glm::vec3 rotateAxis);
    int mVertNum;
    ~model();
};


#endif //JIARUI_MPM_MODEL_H
