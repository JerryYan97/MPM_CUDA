//
// Created by jiaruiyan on 5/26/21.
//

#include "MPMSimulator.h"
#include "MPMCUDA.cuh"
#include "mesh_query.h"
#include "../model/model.h"
#include <random>
#include <iostream>

MPMSimulator::MPMSimulator(double gap, unsigned int nodeNumDim, unsigned int particleNumPerCell,
                           std::string &sampleModelPath) {
    mGrid.h = gap;
    mGrid.nodeNumDim = nodeNumDim;
    mGrid.originCorner = {0.0, 0.0, 0.0};
    model mModel(sampleModelPath, 1.f, false);
    mModel.setTransformation(glm::vec3(1.f),
                             glm::vec3(10.f, 10.f, 10.f),
                             0.f,
                             glm::vec3(1.f, 0.f, 0.f));

    // Check the obj bounding box is within the grid.
    if (mModel.mLowerBound[0] < 0.f || mModel.mLowerBound[1] < 0.f || mModel.mLowerBound[2] < 0.f){
        throw "ERROR: OBJ lower bound is smaller than grid's origin.";
    }
    float gridUpperBound = (nodeNumDim - 1) * gap;
    if (mModel.mUpperBound[0] > gridUpperBound || mModel.mUpperBound[1] > gridUpperBound || mModel.mUpperBound[2] > gridUpperBound){
        throw "ERROR: OBJ upper bound is out of the grid.";
    }

    // Get lower and upper grid index
    int lower_x_idx = int(mModel.mLowerBound[0] / gap);
    int lower_y_idx = int(mModel.mLowerBound[1] / gap);
    int lower_z_idx = int(mModel.mLowerBound[2] / gap);
    int upper_x_idx = int(mModel.mUpperBound[0] / gap);
    int upper_y_idx = int(mModel.mUpperBound[1] / gap);
    int upper_z_idx = int(mModel.mUpperBound[2] / gap);

    // Put random particles into every grids between the lower and upper grid.
    MeshObject* mMOBJ = construct_mesh_object(mModel.mQMVertData.size() / 3,
                                              mModel.mQMVertData.data(),
                                              mModel.mQMIndData.size() / 3,
                                              mModel.mQMIndData.data());

    glm::mat4 modelInv = glm::inverse(mModel.mModelMat);
    for (int ix = lower_x_idx; ix < upper_x_idx; ++ix) {
        for (int iy = lower_y_idx; iy < upper_y_idx; ++iy) {
            for (int iz = lower_z_idx; iz < upper_z_idx; ++iz) {
                for (int ip = 0; ip < particleNumPerCell; ++ip) {
                    float rx = float(rand()) / float(RAND_MAX);
                    float ry = float(rand()) / float(RAND_MAX);
                    float rz = float(rand()) / float(RAND_MAX);
                    float node_x = ix * gap + rx * gap;
                    float node_y = iy * gap + ry * gap;
                    float node_z = iz * gap + rz * gap;
                    glm::vec4 localPt = modelInv * glm::vec4(node_x, node_y, node_z, 1.f);
                    double pt[3] = {localPt[0], localPt[1], localPt[2]};
                    if (point_inside_mesh(pt, mMOBJ)){
                        mParticles.particlePosVec.push_back(node_x);
                        mParticles.particlePosVec.push_back(node_y);
                        mParticles.particlePosVec.push_back(node_z);
                    }
                }
            }
        }
    }
    mParticles.particleNum = mParticles.particlePosVec.size() / 3;
    mParticles.particlePosVec.resize(mParticles.particleNum * 3);
    mParticles.particleMassVec.resize(mParticles.particleNum);
    mParticles.particleVelVec.resize(mParticles.particleNum * 3);
    destroy_mesh_object(mMOBJ);
}
