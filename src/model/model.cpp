//
// Created by jiaruiyan on 5/26/21.
//

#include "model.h"

#include <iostream>
#include <glad/glad.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

model::model(std::string &path, float modelSize, bool usedGL) {
    // Read in .obj mesh.
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig reader_config;

    if (!reader.ParseFromFile(path)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& shapes = reader.GetShapes();
    auto& attrib = reader.GetAttrib();
    int vert_num = shapes[0].mesh.num_face_vertices.size() * 3;
    mVertNum = vert_num;
    if (usedGL){
        mGLVertData.resize(vert_num * 6);
        mGLIndices.resize(vert_num);
    }
    mLowerBound = glm::vec3(100000.f, 100000.f, 100000.f);
    mUpperBound = glm::vec3(-100000.f, -100000.f, -100000.f);
    for (size_t s = 0; s < shapes.size(); s++) {
        int vert_idx = 0;
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex.
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                if (usedGL){
                    tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                    tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                    tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                    mGLVertData[vert_idx * 6] = vx;
                    mGLVertData[vert_idx * 6 + 1] = vy;
                    mGLVertData[vert_idx * 6 + 2] = vz;
                    mGLVertData[vert_idx * 6 + 3] = nx;
                    mGLVertData[vert_idx * 6 + 4] = ny;
                    mGLVertData[vert_idx * 6 + 5] = nz;
                }

                glm::vec4 global_pos = mModelMat * glm::vec4(vx, vy, vz, 1.f);
                mLowerBound[0] = glm::min(global_pos[0], mLowerBound[0]);
                mLowerBound[1] = glm::min(global_pos[1], mLowerBound[1]);
                mLowerBound[2] = glm::min(global_pos[2], mLowerBound[2]);
                mUpperBound[0] = glm::max(global_pos[0], mUpperBound[0]);
                mUpperBound[1] = glm::max(global_pos[1], mUpperBound[1]);
                mUpperBound[2] = glm::max(global_pos[2], mUpperBound[2]);

                if (usedGL){
                    mGLIndices[vert_idx] = vert_idx;
                    vert_idx++;
                }
            }
            index_offset += fv;
        }
    }

    // Init matrix.
    mModelMat = glm::mat4(1.f);
    mModelMat = glm::translate(mModelMat, glm::vec3(0.f, 0.f, 0.f));
    mModelMat = glm::scale(mModelMat, glm::vec3(modelSize));
    mModelMat = glm::rotate(mModelMat, glm::radians(0.f), glm::vec3(1.f, 0.f, 0.f));

    mNormalMat = glm::mat3(1.0f);
    mNormalMat[0, 0] = mModelMat[0, 0];
    mNormalMat[0, 1] = mModelMat[0, 1];
    mNormalMat[0, 2] = mModelMat[0, 2];
    mNormalMat[1, 0] = mModelMat[1, 0];
    mNormalMat[1, 1] = mModelMat[1, 1];
    mNormalMat[1, 2] = mModelMat[1, 2];
    mNormalMat[2, 0] = mModelMat[2, 0];
    mNormalMat[2, 1] = mModelMat[2, 1];
    mNormalMat[2, 2] = mModelMat[2, 2];
    mNormalMat = glm::transpose(glm::inverse(mNormalMat));

    if (!usedGL){
        mQMIndData.resize(shapes[0].mesh.indices.size());
        for (int i = 0; i < shapes[0].mesh.indices.size(); ++i) {
            mQMIndData[i] = shapes[0].mesh.indices[i].vertex_index;
        }
        mQmVertData = std::vector<double>(attrib.vertices.begin(), attrib.vertices.end());
    }
}

// Line boundary input position is in the world space.
model::model(const float *lowerBound, const float *upperBound) {
    // Init raw data
    float p1[3] = {lowerBound[0], lowerBound[1], lowerBound[2]};
    float p2[3] = {lowerBound[0], lowerBound[1], upperBound[2]};
    float p3[3] = {upperBound[0], lowerBound[1], upperBound[2]};
    float p4[3] = {upperBound[0], lowerBound[1], lowerBound[2]};
    float p5[3] = {lowerBound[0], upperBound[1], lowerBound[2]};
    float p6[3] = {lowerBound[0], upperBound[1], upperBound[2]};
    float p7[3] = {upperBound[0], upperBound[1], lowerBound[2]};
    float p8[3] = {upperBound[0], upperBound[1], upperBound[2]};

    // Transfer to GL format
    mGLVertData.resize(3 * 8);
    mGLVertData[0] = p1[0]; mGLVertData[1] = p1[1]; mGLVertData[2] = p1[2];
    mGLVertData[3] = p2[0]; mGLVertData[4] = p2[1]; mGLVertData[5] = p2[2];
    mGLVertData[6] = p3[0]; mGLVertData[7] = p3[1]; mGLVertData[8] = p3[2];
    mGLVertData[9] = p4[0]; mGLVertData[10] = p4[1]; mGLVertData[11] = p4[2];
    mGLVertData[12] = p5[0]; mGLVertData[13] = p5[1]; mGLVertData[14] = p5[2];
    mGLVertData[15] = p6[0]; mGLVertData[16] = p6[1]; mGLVertData[17] = p6[2];
    mGLVertData[18] = p7[0]; mGLVertData[19] = p7[1]; mGLVertData[20] = p7[2];
    mGLVertData[21] = p8[0]; mGLVertData[22] = p8[1]; mGLVertData[23] = p8[2];

    // Init Indices
    mGLIndices.resize(12 * 2);

    mGLIndices[0] = 0; mGLIndices[1] = 1;
    mGLIndices[2] = 1; mGLIndices[3] = 2;
    mGLIndices[4] = 2; mGLIndices[5] = 3;
    mGLIndices[6] = 3; mGLIndices[7] = 0;

    mGLIndices[8] = 4; mGLIndices[9] = 5;
    mGLIndices[10] = 5; mGLIndices[11] = 7;
    mGLIndices[12] = 4; mGLIndices[13] = 6;
    mGLIndices[14] = 6; mGLIndices[15] = 7;

    mGLIndices[16] = 0; mGLIndices[17] = 4;
    mGLIndices[18] = 1; mGLIndices[19] = 5;
    mGLIndices[20] = 2; mGLIndices[21] = 7;
    mGLIndices[22] = 3; mGLIndices[23] = 6;

    mUseInGL = true;
    mModelMat = glm::mat4(1.f);
    mLowerBound = glm::vec3(mModelMat * glm::vec4(lowerBound[0], lowerBound[1], lowerBound[2], 1.f));
    mUpperBound = glm::vec3(mModelMat * glm::vec4(upperBound[0], upperBound[1], upperBound[2], 1.f));
    mVertNum = 8;
}



model::~model() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

void model::transGRAM() {
    // Transfer to GRAM.
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * mVertNum * 6, mGLVertData.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mVertNum, mGLIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void model::transLinesGRAM() {
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * mVertNum * 3, mGLVertData.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mGLIndices.size(), mGLIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// NOTE: It only works for data format with both pos(vec3) and normal(vec3).
void model::setTransformation(glm::vec3 scale, glm::vec3 translation, float rotateDegree, glm::vec3 rotateAxis) {
    // Reset matrix.
    mModelMat = glm::mat4(1.f);
    mModelMat = glm::translate(mModelMat, translation);
    mModelMat = glm::scale(mModelMat, scale);
    mModelMat = glm::rotate(mModelMat, glm::radians(rotateDegree), rotateAxis);
    // mLowerBound = glm::vec3(mModelMat * glm::vec4(mLowerBound, 1.f));
    // mUpperBound = glm::vec3(mModelMat * glm::vec4(mUpperBound, 1.f));

    mLowerBound = glm::vec3(100000.f, 100000.f, 100000.f);
    mUpperBound = glm::vec3(-100000.f, -100000.f, -100000.f);
    for (int i = 0; i < mQmVertData.size() / 3; ++i) {
        glm::vec4 global_pos = mModelMat * glm::vec4(mQmVertData[i * 3], mQmVertData[i * 3 + 1],
                                                     mQmVertData[i * 3 + 2], 1.f);
        mLowerBound[0] = glm::min(global_pos[0], mLowerBound[0]);
        mLowerBound[1] = glm::min(global_pos[1], mLowerBound[1]);
        mLowerBound[2] = glm::min(global_pos[2], mLowerBound[2]);
        mUpperBound[0] = glm::max(global_pos[0], mUpperBound[0]);
        mUpperBound[1] = glm::max(global_pos[1], mUpperBound[1]);
        mUpperBound[2] = glm::max(global_pos[2], mUpperBound[2]);
    }

    mNormalMat = glm::mat3(1.0f);
    mNormalMat[0, 0] = mModelMat[0, 0];
    mNormalMat[0, 1] = mModelMat[0, 1];
    mNormalMat[0, 2] = mModelMat[0, 2];
    mNormalMat[1, 0] = mModelMat[1, 0];
    mNormalMat[1, 1] = mModelMat[1, 1];
    mNormalMat[1, 2] = mModelMat[1, 2];
    mNormalMat[2, 0] = mModelMat[2, 0];
    mNormalMat[2, 1] = mModelMat[2, 1];
    mNormalMat[2, 2] = mModelMat[2, 2];
    mNormalMat = glm::transpose(glm::inverse(mNormalMat));

}






