//
// Created by jiaruiyan on 5/26/21.
//

#include "MPMSimulator.cuh"
#include "mesh_query.h"
#include "../model/model.h"
#include <random>

void MPMSimulator::initParticles(std::vector<double> &volVec) {
    cudaError_t err = cudaSuccess;
    // Init the sampling result.
    mParticles.particleMassVec.resize(mParticles.particleNum);
    mParticles.particleVelVec.resize(mParticles.particleNum * 3);
    std::fill(mParticles.particleVelVec.begin(), mParticles.particleVelVec.end(), 0.0);
    // Init the volume, mass result.
    mParticles.particleVolVec.resize(mParticles.particleNum);
    for (int i = 0; i < volVec.size(); ++i) {
        double pVol = volVec[i] / double(mParticles.particleNumDiv[i]);
        if (i == 0){
            std::fill(mParticles.particleVolVec.begin(),
                      mParticles.particleVolVec.begin() + mParticles.particleNumDiv[0],
                      pVol);
            std::fill(mParticles.particleMassVec.begin(),
                      mParticles.particleMassVec.end() + mParticles.particleNumDiv[0],
                      pVol * mParticles.mMaterialVec[i].mDensity);
        }
        else{
            int beginOffset = 0;
            for (int j = 0; j < i; ++j) {
                beginOffset += mParticles.particleNumDiv[j];
            }
            std::fill(mParticles.particleVolVec.begin() + beginOffset,
                      mParticles.particleVolVec.begin() + beginOffset + mParticles.particleNumDiv[i],
                      pVol);
            std::fill(mParticles.particleMassVec.begin() + beginOffset,
                      mParticles.particleMassVec.end() + beginOffset + mParticles.particleNumDiv[i],
                      pVol * mParticles.mMaterialVec[i].mDensity);
        }
    }
    // Init the deformation gradient.
    std::vector<double> tmpDeformationGradientVec(mParticles.particleNum * 9, 0.0);
    for (int i = 0; i < mParticles.particleNum; ++i) {
        tmpDeformationGradientVec[9 * i] = 1.0;
        tmpDeformationGradientVec[9 * i + 4] = 1.0;
        tmpDeformationGradientVec[9 * i + 8] = 1.0;
    }

    mParticles.posVecByteSize = mParticles.particleNum * 3 * sizeof(double);
    mParticles.massVecByteSize = mParticles.particleNum * sizeof(double);
    mParticles.velVecByteSize = mParticles.particleNum * 3 * sizeof(double);
    mParticles.volVecByteSize = mParticles.particleNum * sizeof(double);
    mParticles.dgVecByteSize = mParticles.particleNum * 9 * sizeof(double); // 11, 12, 13, 21, 22, 23, 31, 32, 33.

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

    err = cudaMalloc((void **)&mParticles.pVolVecGRAM, mParticles.volVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles volume error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(mParticles.pVolVecGRAM, mParticles.particleVolVec.data(),
                     mParticles.volVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init particles volume error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaMalloc((void **)&mParticles.pDeformationGradientGRAM, mParticles.dgVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(mParticles.pDeformationGradientGRAM, tmpDeformationGradientVec.data(),
                     mParticles.dgVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}


void MPMSimulator::initGrid(double gap, unsigned int nodeNumDim) {
    // Init grid
    cudaError_t err = cudaSuccess;
    mGrid.h = gap;
    mGrid.nodeNumDim = nodeNumDim;
    mGrid.originCorner = {0.0, 0.0, 0.0};
    mGrid.massVecByteSize = mGrid.nodeNumDim * mGrid.nodeNumDim * mGrid.nodeNumDim * sizeof(double);
    mGrid.velVecByteSize = mGrid.nodeNumDim * mGrid.nodeNumDim * mGrid.nodeNumDim * sizeof(double) * 3;
    mGrid.forceVecByteSize = mGrid.nodeNumDim * mGrid.nodeNumDim * mGrid.nodeNumDim * sizeof(double) * 3;

    std::cout << "Grid mass uses GRAM and RAM:" << float(mGrid.massVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Grid vel uses GRAM and RAM:" << float(mGrid.velVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Grid force uses GRAM and RAM:" << float(mGrid.forceVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;

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
    err = cudaMalloc((void **)&mGrid.nodeForceVec, mGrid.forceVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate grid force error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void MPMSimulator::showMemUsage() {
    // Show memory usage.
    std::cout << "Particles mass GRAM and RAM:" << float(mParticles.massVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles pos GRAM and RAM:" << float(mParticles.posVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles vel GRAM and RAM:" << float(mParticles.velVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;

    std::cout << "Sampled particle number:" << mParticles.particleNum << std::endl;
}


MPMSimulator::MPMSimulator(double gap, double dt, unsigned int nodeNumDim, unsigned int particleNumPerCell,
                           std::string &sampleModelPath) {
    // Init info.
    this->dt = dt;
    t = 0.0;
    ext_gravity = 0.0;
    FixedCorotatedMaterial mMaterial(0.01e9, 0.49, 1.1);
    mParticles.mMaterialVec.push_back(mMaterial);
    current_frame = 0;
    current_time = 0.0;

    initGrid(gap, nodeNumDim);

    // Load and sample model.
    cudaError_t err = cudaSuccess;
    model mModel(sampleModelPath, 1.f, false);
    mModel.setTransformation(glm::vec3(1.f),
                             glm::vec3(5.f, 5.f, 5.f),
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
    std::vector<double> volVec;
    volVec.push_back(calVolmue(sampleModelPath));
    mParticles.particleNumDiv.push_back(mParticles.particleNum);

    destroy_mesh_object(mMOBJ);
    initParticles(volVec);

    std::cout << "Sampled model lower bound:"
              << mModel.mLowerBound[0] << " "
              << mModel.mLowerBound[1] << " "
              << mModel.mLowerBound[2]<< std::endl;

    std::cout << "Sampled model upper bound:"
              << mModel.mUpperBound[0] << " "
              << mModel.mUpperBound[1] << " "
              << mModel.mUpperBound[2]<< std::endl;

    showMemUsage();
}


MPMSimulator::MPMSimulator(double gap, double dt, unsigned int nodeNumDim, unsigned int particleNumPerCell,
                           std::string &sampleModelPath1, std::string &sampleModelPath2) {
    // Init info.
    this->dt = dt;
    t = 0.0;
    ext_gravity = 0.0;
    FixedCorotatedMaterial mMaterial1(1e3, 0.2, 1.1);
    FixedCorotatedMaterial mMaterial2(1e3, 0.2, 1.1);
    mParticles.mMaterialVec.push_back(mMaterial1);
    mParticles.mMaterialVec.push_back(mMaterial2);
    current_frame = 0;
    current_time = 0.0;

    initGrid(gap, nodeNumDim);

    // Load and sample model:
    cudaError_t err = cudaSuccess;
    model mModel1(sampleModelPath1, 1.f, false);
    mModel1.setTransformation(glm::vec3(1.f),
                              glm::vec3(6.0f, 5.f, 5.f),
                              0.f,
                              glm::vec3(1.f, 0.f, 0.f));
    model mModel2(sampleModelPath1, 1.f, false);
    mModel2.setTransformation(glm::vec3(1.f),
                              glm::vec3(3.0f, 5.f, 5.f),
                              0.f,
                              glm::vec3(1.f, 0.f, 0.f));

    // Check the obj bounding box is within the grid.
    float gridUpperBound = (nodeNumDim - 1) * gap;
    if (std::min(mModel1.mLowerBound[0], mModel2.mLowerBound[0]) < 0.f ||
        std::min(mModel1.mLowerBound[1], mModel2.mLowerBound[1]) < 0.f ||
        std::min(mModel1.mLowerBound[2], mModel2.mLowerBound[2]) < 0.f)
    {
        std::cerr << "ERROR: OBJ lower bound is smaller than grid's origin." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    if (std::max(mModel1.mUpperBound[0], mModel2.mUpperBound[0]) > gridUpperBound ||
        std::max(mModel1.mUpperBound[1], mModel2.mUpperBound[1]) > gridUpperBound ||
        std::max(mModel1.mUpperBound[2], mModel2.mUpperBound[2]) > gridUpperBound)
    {
        std::cerr << "ERROR: OBJ upper bound is out of the grid." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Get lower and upper grid index of model 1.
    int lower1_x_idx = int(mModel1.mLowerBound[0] / gap);
    int lower1_y_idx = int(mModel1.mLowerBound[1] / gap);
    int lower1_z_idx = int(mModel1.mLowerBound[2] / gap);
    int upper1_x_idx = int(mModel1.mUpperBound[0] / gap);
    int upper1_y_idx = int(mModel1.mUpperBound[1] / gap);
    int upper1_z_idx = int(mModel1.mUpperBound[2] / gap);

    // Put random particles into every grids between the lower and upper grid.
    MeshObject* mMOBJ1 = construct_mesh_object(mModel1.mQMVertData.size() / 3,
                                              mModel1.mQMVertData.data(),
                                              mModel1.mQMIndData.size() / 3,
                                              mModel1.mQMIndData.data());

    glm::mat4 model1Inv = glm::inverse(mModel1.mModelMat);
    for (int ix = lower1_x_idx; ix < upper1_x_idx; ++ix) {
        for (int iy = lower1_y_idx; iy < upper1_y_idx; ++iy) {
            for (int iz = lower1_z_idx; iz < upper1_z_idx; ++iz) {
                for (int ip = 0; ip < particleNumPerCell; ++ip) {
                    float rx = float(rand()) / float(RAND_MAX);
                    float ry = float(rand()) / float(RAND_MAX);
                    float rz = float(rand()) / float(RAND_MAX);
                    float node_x = ix * gap + rx * gap;
                    float node_y = iy * gap + ry * gap;
                    float node_z = iz * gap + rz * gap;
                    glm::vec4 localPt = model1Inv * glm::vec4(node_x, node_y, node_z, 1.f);
                    double pt[3] = {localPt[0], localPt[1], localPt[2]};
                    if (point_inside_mesh(pt, mMOBJ1)){
                        if (ix == (upper1_x_idx - 1)){
                            int b1_r_pidx = mParticles.particlePosVec.size() / 3;
                            idx_vec.push_back(b1_r_pidx);
                        }
                        mParticles.particlePosVec.push_back(node_x);
                        mParticles.particlePosVec.push_back(node_y);
                        mParticles.particlePosVec.push_back(node_z);
                    }
                }
            }
        }
    }

    /* 51984
    std::cout << "The box right idx are:" << std::endl;
    for (int i = 0; i < idx_vec.size(); ++i) {
        std::cout << idx_vec[i] << std::endl;
    }
    */

    int model1ParticleNum = mParticles.particlePosVec.size() / 3;

    // Get lower and upper grid index of model 2.
    int lower2_x_idx = int(mModel2.mLowerBound[0] / gap);
    int lower2_y_idx = int(mModel2.mLowerBound[1] / gap);
    int lower2_z_idx = int(mModel2.mLowerBound[2] / gap);
    int upper2_x_idx = int(mModel2.mUpperBound[0] / gap);
    int upper2_y_idx = int(mModel2.mUpperBound[1] / gap);
    int upper2_z_idx = int(mModel2.mUpperBound[2] / gap);

    // Put random particles into every grids between the lower and upper grid.
    MeshObject* mMOBJ2 = construct_mesh_object(mModel2.mQMVertData.size() / 3,
                                               mModel2.mQMVertData.data(),
                                               mModel2.mQMIndData.size() / 3,
                                               mModel2.mQMIndData.data());

    glm::mat4 model2Inv = glm::inverse(mModel2.mModelMat);
    for (int ix = lower2_x_idx; ix < upper2_x_idx; ++ix) {
        for (int iy = lower2_y_idx; iy < upper2_y_idx; ++iy) {
            for (int iz = lower2_z_idx; iz < upper2_z_idx; ++iz) {
                for (int ip = 0; ip < particleNumPerCell; ++ip) {
                    float rx = float(rand()) / float(RAND_MAX);
                    float ry = float(rand()) / float(RAND_MAX);
                    float rz = float(rand()) / float(RAND_MAX);
                    float node_x = ix * gap + rx * gap;
                    float node_y = iy * gap + ry * gap;
                    float node_z = iz * gap + rz * gap;
                    glm::vec4 localPt = model2Inv * glm::vec4(node_x, node_y, node_z, 1.f);
                    double pt[3] = {localPt[0], localPt[1], localPt[2]};
                    if (point_inside_mesh(pt, mMOBJ2)){
                        mParticles.particlePosVec.push_back(node_x);
                        mParticles.particlePosVec.push_back(node_y);
                        mParticles.particlePosVec.push_back(node_z);
                    }
                }
            }
        }
    }

    mParticles.particleNum = mParticles.particlePosVec.size() / 3;
    mParticles.particleNumDiv.push_back(model1ParticleNum);
    mParticles.particleNumDiv.push_back(mParticles.particleNum - model1ParticleNum);

    destroy_mesh_object(mMOBJ1);
    destroy_mesh_object(mMOBJ2);
    std::vector<double> volVec;
    volVec.push_back(calVolmue(sampleModelPath1));
    volVec.push_back(calVolmue(sampleModelPath2));
    initParticles(volVec);

#ifdef DEBUG
    // Check particle volume.
    double lastVol = mParticles.particleVolVec[0];
    double lastMass = mParticles.particleMassVec[0];
    std::cout << "vol0:" << lastVol << std::endl;
    std::cout << "mass0:" << lastMass << std::endl;
    for (int i = 0; i < mParticles.particleNum; ++i) {
        if (lastVol != mParticles.particleVolVec[i]){
            lastVol = mParticles.particleVolVec[i];
            std::cout << "vol" << i+1 << ":" << lastVol << std::endl;
        }
        if (lastMass != mParticles.particleMassVec[i]){
            lastMass = mParticles.particleMassVec[i];
            std::cout << "mass" << i+1 << ":" << lastMass << std::endl;
        }
    }
#endif

    std::cout << "Sampled model lower bound:"
              << std::min(mModel1.mLowerBound[0], mModel2.mLowerBound[0]) << " "
              << std::min(mModel1.mLowerBound[1], mModel2.mLowerBound[1]) << " "
              << std::min(mModel1.mLowerBound[2], mModel2.mLowerBound[2]) << std::endl;

    std::cout << "Sampled model upper bound:"
              << std::max(mModel1.mUpperBound[0], mModel2.mUpperBound[0]) << " "
              << std::max(mModel1.mUpperBound[1], mModel2.mUpperBound[1]) << " "
              << std::max(mModel1.mUpperBound[2], mModel2.mUpperBound[2]) << std::endl;

    min_bound_x = std::min(mModel1.mLowerBound[0], mModel2.mLowerBound[0]) - DBL_EPSILON;
    min_bound_y = std::min(mModel1.mLowerBound[1], mModel2.mLowerBound[1]) - DBL_EPSILON;
    min_bound_z = std::min(mModel1.mLowerBound[2], mModel2.mLowerBound[2]) - DBL_EPSILON;
    max_bound_x = std::max(mModel1.mUpperBound[0], mModel2.mUpperBound[0]) + DBL_EPSILON;
    max_bound_y = std::max(mModel1.mUpperBound[1], mModel2.mUpperBound[1]) + DBL_EPSILON;
    max_bound_z = std::max(mModel1.mUpperBound[2], mModel2.mUpperBound[2]) + DBL_EPSILON;

    showMemUsage();
}


MPMSimulator::~MPMSimulator() {
    cudaError_t err = cudaSuccess;

    // Free GRAM associated with Grid.
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

    err = cudaFree(mGrid.nodeForceVec);
    if (err != cudaSuccess){
        std::cerr << "Free grid Force error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Free GRAM associated with Particles.
    err = cudaFree(mParticles.pPosVecGRAM);
    if (err != cudaSuccess){
        std::cerr << "Free particle position error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaFree(mParticles.pVelVecGRAM);
    if (err != cudaSuccess){
        std::cerr << "Free particle velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaFree(mParticles.pVolVecGRAM);
    if (err != cudaSuccess){
        std::cerr << "Free particle volume error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void MPMSimulator::setVel(std::vector<double> &particleVelVec) {
    mParticles.particleVelVec = particleVelVec;
    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(mParticles.pVelVecGRAM,mParticles.particleVelVec.data(),
                     mParticles.velVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Set Velocity memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

double MPMSimulator::calVolmue(std::string &place) {
    return 2.0 * 2.0 *  2.0; // For cube
}
