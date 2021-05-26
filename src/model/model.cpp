//
// Created by jiaruiyan on 5/26/21.
//

#include "model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <iostream>
#include <glad/glad.h>

model::model(std::string &path) {

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
    mVertData.resize(vert_num * 6);
    mIndices.resize(vert_num);
    for (size_t s = 0; s < shapes.size(); s++) {
        int vert_idx = 0;
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                mVertData[vert_idx * 6] = vx;
                mVertData[vert_idx * 6 + 1] = vy;
                mVertData[vert_idx * 6 + 2] = vz;
                mVertData[vert_idx * 6 + 3] = nx;
                mVertData[vert_idx * 6 + 4] = ny;
                mVertData[vert_idx * 6 + 5] = nz;
                mIndices[vert_idx] = vert_idx;
                vert_idx++;
            }
            index_offset += fv;
        }
    }

    // Init matrix.
    mModelMat = glm::rotate(glm::mat4(1.f), glm::radians(0.f), glm::vec3(1.f, 0.f, 0.f));
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

    // Transfer to GRAM.
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vert_num * 6, mVertData.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * vert_num, mIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

model::~model() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

