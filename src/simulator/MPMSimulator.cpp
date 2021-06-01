//
// Created by jiaruiyan on 5/26/21.
//

#include "MPMSimulator.cuh"
#include "mesh_query.h"
#include "../model/model.h"
#include <random>

MPMSimulator::MPMSimulator(double gap, double dt, unsigned int nodeNumDim, unsigned int particleNumPerCell,
                           std::string &sampleModelPath) {
    this->dt = dt;
    t = 0.0;
    // Init grid
    cudaError_t err = cudaSuccess;
    mGrid.h = gap;
    mGrid.nodeNumDim = nodeNumDim;
    mGrid.originCorner = {0.0, 0.0, 0.0};
    mGrid.massVecByteSize = mGrid.nodeNumDim * mGrid.nodeNumDim * mGrid.nodeNumDim * sizeof(double);
    mGrid.velVecByteSize = mGrid.nodeNumDim * mGrid.nodeNumDim * mGrid.nodeNumDim * sizeof(double) * 3;
    std::cout << "Grid mass uses GRAM and RAM:" << float(mGrid.massVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Grid vel uses GRAM and RAM:" << float(mGrid.velVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    err = cudaMalloc((void **)&mGrid.nodeMassVec, mGrid.massVecByteSize);
    if(err != cudaSuccess){
        std::cerr << "Allocate grid mass error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMalloc((void **)&mGrid.nodeVelVec, mGrid.velVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate grid velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    model mModel(sampleModelPath, 1.f, false);
    mModel.setTransformation(glm::vec3(1.f),
                             glm::vec3(5.f, 40.f, 5.f),
                             0.f,
                             glm::vec3(1.f, 0.f, 0.f));

    // Check the obj bounding box is within the grid.
    if (mModel.mLowerBound[0] < 0.f || mModel.mLowerBound[1] < 0.f || mModel.mLowerBound[2] < 0.f){
        std::cerr << "ERROR: OBJ lower bound is smaller than grid's origin." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    float gridUpperBound = (nodeNumDim - 1) * gap;
    if (mModel.mUpperBound[0] > gridUpperBound || mModel.mUpperBound[1] > gridUpperBound || mModel.mUpperBound[2] > gridUpperBound){
        std::cerr << "ERROR: OBJ upper bound is out of the grid." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
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
    mParticles.particleMassVec.resize(mParticles.particleNum);
    std::fill(mParticles.particleMassVec.begin(), mParticles.particleMassVec.end(), 1.0);
    mParticles.particleVelVec.resize(mParticles.particleNum * 3);
    std::fill(mParticles.particleVelVec.begin(), mParticles.particleVelVec.end(), 0.0);
    destroy_mesh_object(mMOBJ);

    mParticles.posVecByteSize = mParticles.particleNum * 3 * sizeof(double);
    mParticles.massVecByteSize = mParticles.particleNum * sizeof(double);
    mParticles.velVecByteSize = mParticles.particleNum * 3 * sizeof(double);
    err = cudaMalloc((void **)&mParticles.pPosVecGRAM, mParticles.posVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles pos error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(mParticles.pPosVecGRAM, mParticles.particlePosVec.data(),
               mParticles.posVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init particles pos error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaMalloc((void **)&mParticles.pMassVecGRAM, mParticles.massVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles mass error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(mParticles.pMassVecGRAM, mParticles.particleMassVec.data(),
                     mParticles.massVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init particles mass error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaMalloc((void **)&mParticles.pVelVecGRAM, mParticles.velVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(mParticles.pVelVecGRAM, mParticles.particleVelVec.data(),
                     mParticles.velVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init particles velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Show memory usage.
    std::cout << "Particles mass GRAM and RAM:" << float(mParticles.massVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles pos GRAM and RAM:" << float(mParticles.posVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles vel GRAM and RAM:" << float(mParticles.velVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;

    std::cout << "Sampled particle number:" << mParticles.particleNum << std::endl;

    std::cout << "Sampled model lower bound:"
              << mModel.mLowerBound[0] << " "
              << mModel.mLowerBound[1] << " "
              << mModel.mLowerBound[2]<< std::endl;

    std::cout << "Sampled model upper bound:"
              << mModel.mUpperBound[0] << " "
              << mModel.mUpperBound[1] << " "
              << mModel.mUpperBound[2]<< std::endl;
}

MPMSimulator::~MPMSimulator() {
    cudaError_t err = cudaSuccess;

    err = cudaFree(mGrid.nodeMassVec);
    if(err != cudaSuccess){
        std::cerr << "Free grid mass error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaFree(mGrid.nodeVelVec);
    if (err != cudaSuccess){
        std::cerr << "Free grid velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
