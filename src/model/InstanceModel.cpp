//
// Created by jiaruiyan on 5/26/21.
//

#include "InstanceModel.h"
#include <glad/glad.h>
#include <cstring>
#include <iostream>

InstanceModel::InstanceModel(std::string &path, std::vector<float>& posVec) : model(path) {
    this->transGRAM();
    mInstanceNum = posVec.size() / 3;
    glGenBuffers(1, &mInstancedArrayBO);
    glBindBuffer(GL_ARRAY_BUFFER, mInstancedArrayBO);
    glBufferData(GL_ARRAY_BUFFER, posVec.size() * sizeof(float), posVec.data(), GL_STATIC_DRAW);

    glBindVertexArray(VAO);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribDivisor(2, 1);
    glBindVertexArray(0);
}

void InstanceModel::updateInstanceModel(std::vector<float> &posVec) {
    if (posVec.size()/3 != mInstanceNum){
        throw "ERROR: Updated instance number is inequal to the original number.";
    }
    glBindBuffer(GL_ARRAY_BUFFER, mInstancedArrayBO);
    void* ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    std::memcpy(ptr, posVec.data(), posVec.size() * sizeof(float));
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

InstanceModel::~InstanceModel() {
    glDeleteBuffers(1, &mInstancedArrayBO);
}
