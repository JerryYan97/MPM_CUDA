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
    bool mUseInGL{};
public:
    unsigned int VBO{}, EBO{}, VAO{};
    glm::mat4 mModelMat{};
    glm::mat3 mNormalMat{};

    std::vector<float> mGLVertData;
    std::vector<unsigned int> mGLIndices;
    std::vector<int> mQMIndData; // Query mesh indices: For MPM sampling.
    std::vector<double> mQmVertData;
    std::vector<unsigned int> mVertLength; // Support multiple models.

    glm::vec3 mUpperBound{}; // Upper bound in the world space.
    glm::vec3 mLowerBound{}; // Lower bound in the world space.

    explicit model(std::string &path, float modelSize, bool usedGL); // For normal mesh;
    model(const float * lowerBound, const float * upperBound); // For boundary lines;

    void transGRAM(); // For normal mesh;
    void transLinesGRAM(); // For boundary lines;

    void setTransformation(glm::vec3 scale, glm::vec3 translation, float rotateDegree, glm::vec3 rotateAxis);
    int mVertNum{}; // NOTE: This is the number of vertices for OpenGL data format instead of distinctive points' number in this model.
    ~model();
};


#endif //JIARUI_MPM_MODEL_H
