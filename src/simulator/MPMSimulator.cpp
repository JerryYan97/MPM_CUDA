//
// Created by jiaruiyan on 5/26/21.
//

#include "MPMSimulator.cuh"
#include "mesh_query.h"
#include "../model/model.h"
#include <random>

void MPMSimulator::initParticles(ParticleGroup& initPG, ObjInitInfo& initObjInfo) {
    cudaError_t err = cudaSuccess;
    // Init the velocity.
    std::vector<double> particleVelVec(initPG.particleNum * 3, 0.0);
    for (int i = 0; i < initPG.particleNum; ++i) {
        particleVelVec[i * 3] = initObjInfo.initVel[0];
        particleVelVec[i * 3 + 1] = initObjInfo.initVel[1];
        particleVelVec[i * 3 + 2] = initObjInfo.initVel[2];
    }

    // Init the deformation gradient.
    std::vector<double> tmpDeformationGradientVec(initPG.particleNum * 9, 0.0);
    for (int i = 0; i < initPG.particleNum; ++i) {
        tmpDeformationGradientVec[9 * i] = 1.0;
        tmpDeformationGradientVec[9 * i + 4] = 1.0;
        tmpDeformationGradientVec[9 * i + 8] = 1.0;
    }

    initPG.posVecByteSize = initPG.particleNum * 3 * sizeof(double);
    initPG.velVecByteSize = initPG.particleNum * 3 * sizeof(double);
    initPG.pDgVecByteSize = 0;
    initPG.eDgVecByteSize = initPG.particleNum * 9 * sizeof(double);
    initPG.affineVelVecByteSize = initPG.particleNum * 9 * sizeof(double);
    initPG.dgDiffVecByteSize = initPG.particleNum * sizeof(double);

    err = cudaMalloc((void **)&initPG.pPosVecGRAM, initPG.posVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles pos error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(initPG.pPosVecGRAM, initPG.particlePosVec.data(),
                     initPG.posVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init particles pos error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaMalloc((void **)&initPG.pVelVecGRAM, initPG.velVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(initPG.pVelVecGRAM, particleVelVec.data(),
                     initPG.velVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init particles velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaMalloc((void **)&initPG.pElasiticityDeformationGradientGRAM, initPG.eDgVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles elasiticity deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(initPG.pElasiticityDeformationGradientGRAM, tmpDeformationGradientVec.data(),
                     initPG.eDgVecByteSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Init deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    if (initObjInfo.mMaterial.mType == SNOW){
        initPG.pDgVecByteSize = initPG.particleNum * 9 * sizeof(double);
        err = cudaMalloc((void **)&initPG.pPlasiticityDeformationGradientGRAM, initPG.pDgVecByteSize);
        if (err != cudaSuccess){
            std::cerr << "Allocate particles plasiticity deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
        err = cudaMemcpy(initPG.pPlasiticityDeformationGradientGRAM, tmpDeformationGradientVec.data(),
                         initPG.pDgVecByteSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            std::cerr << "Init deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
    } else if (initObjInfo.mMaterial.mType == WATER){
        initPG.JVecByteSize = initPG.particleNum * sizeof(double);
        err = cudaMalloc((void **)&initPG.pJVecGRAM, initPG.JVecByteSize);
        if (err != cudaSuccess){
            std::cerr << "Allocate water particles' J error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
        std::vector<double> JVecInit(initPG.particleNum, 1.0);
        err = cudaMemcpy(initPG.pJVecGRAM, JVecInit.data(), initPG.JVecByteSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            std::cerr << "Set water particles' J error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
    }

    err = cudaMalloc((void **)&initPG.pAffineVelGRAM, initPG.affineVelVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles affine velocity matrix error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemset(initPG.pAffineVelGRAM, 0, initPG.affineVelVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Set particles affine velocity matrix error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    err = cudaMalloc((void **)&initPG.pDeformationGradientDiffGRAM, initPG.dgDiffVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Allocate particles deformation gradient difference error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemset(initPG.pDeformationGradientDiffGRAM, 0, initPG.dgDiffVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Set particles deformation gradient difference error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void MPMSimulator::initGrid(double gap, unsigned int nodeNumDimX, unsigned int nodeNumDimY, unsigned int nodeNumDimZ) {
    // Init grid
    cudaError_t err = cudaSuccess;
    mGrid.h = gap;
    mGrid.nodeNumDimX = nodeNumDimX;
    mGrid.nodeNumDimY = nodeNumDimY;
    mGrid.nodeNumDimZ = nodeNumDimZ;
    mGrid.originCorner = {0.0, 0.0, 0.0};
    mGrid.massVecByteSize = mGrid.nodeNumDimX * mGrid.nodeNumDimY * mGrid.nodeNumDimZ * sizeof(double);
    mGrid.velVecByteSize = mGrid.nodeNumDimX * mGrid.nodeNumDimY * mGrid.nodeNumDimZ * sizeof(double) * 3;

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
}

void MPMSimulator::showMemUsage() {

    float t_posVecByteSize = 0.f;
    float t_velVecByteSize = 0.f;
    float t_eDgVecByteSize = 0.f;
    float t_pDgVecByteSize = 0.f;
    float t_dgDiffByteSize = 0.f;
    float t_affineVelVecByteSize = 0.f;

    for (int i = 0; i < mParticlesGroupsVec.size(); ++i) {
        t_posVecByteSize += float(mParticlesGroupsVec[i].posVecByteSize);
        t_velVecByteSize += float(mParticlesGroupsVec[i].velVecByteSize);
        t_eDgVecByteSize += float(mParticlesGroupsVec[i].eDgVecByteSize);
        t_pDgVecByteSize += float(mParticlesGroupsVec[i].pDgVecByteSize);
        t_dgDiffByteSize += float(mParticlesGroupsVec[i].dgDiffVecByteSize);
        t_affineVelVecByteSize += float(mParticlesGroupsVec[i].affineVelVecByteSize);
    }

    // Show memory usage.
    std::cout << "Particles pos GRAM:" << t_posVecByteSize / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles vel GRAM:" << t_velVecByteSize / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles eDG GRAM:" << t_eDgVecByteSize / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles pDG GRAM:" << t_pDgVecByteSize / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles dgDiff GRAM:" << t_dgDiffByteSize / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles affine Velocity GRAM:" << t_affineVelVecByteSize / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Particles total GRAM:" << (t_posVecByteSize + t_velVecByteSize + t_eDgVecByteSize + t_pDgVecByteSize +
            t_dgDiffByteSize + t_affineVelVecByteSize) / (1024.f * 1024.f) << "MB" << std::endl;
    std::cout << "Sampled particle number:" << this->totalParticlesNum() << std::endl;
}

MPMSimulator::MPMSimulator(double gap, double max_dt,
                           unsigned int nodeNumDimX, unsigned int nodeNumDimY, unsigned int nodeNumDimZ,
                           unsigned int particleNumPerCell,
                           std::vector<ObjInitInfo> &objInitInfoVec) {

    // Init overall simulator info.
    this->max_dt = max_dt;
    this->adp_dt = max_dt;
    ext_gravity = -9.8;
    current_frame = 0;
    current_time = 0.0;
    initGrid(gap, nodeNumDimX, nodeNumDimY, nodeNumDimZ);
    min_bound_x = 10000.0;
    min_bound_y = 10000.0;
    min_bound_z = 10000.0;
    max_bound_x = -10000.0;
    max_bound_y = -10000.0;
    max_bound_z = -10000.0;

    // Init particles info.
    for (int i = 0; i < objInitInfoVec.size(); ++i) {
        ParticleGroup tmpParticleGroup;
        ObjInitInfo& curInfo = objInitInfoVec[i];
        tmpParticleGroup.mMaterial = curInfo.mMaterial;
        cudaError_t err = cudaSuccess;
        model mModel(curInfo.objPath, 1.f, false);
        mModel.setTransformation(curInfo.initScale,
                                 curInfo.initTranslation,
                                 curInfo.initRotationDegree,
                                 curInfo.initRotationAxis);

        // Check the obj bounding box is within the grid.
        if (mModel.mLowerBound[0] < 0.f || mModel.mLowerBound[1] < 0.f || mModel.mLowerBound[2] < 0.f){
            std::cerr << "ERROR: OBJ lower bound is smaller than grid's origin." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
        float gridUpperBoundX = (nodeNumDimX - 1) * gap;
        float gridUpperBoundY = (nodeNumDimY - 1) * gap;
        float gridUpperBoundZ = (nodeNumDimZ - 1) * gap;
        if (mModel.mUpperBound[0] > gridUpperBoundX ||
            mModel.mUpperBound[1] > gridUpperBoundY ||
            mModel.mUpperBound[2] > gridUpperBoundZ){
            std::cerr << "ERROR: OBJ upper bound is out of the grid." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        // Get lower and upper grid index
        int lower_x_idx = int(mModel.mLowerBound[0] / gap);
        int lower_y_idx = int(mModel.mLowerBound[1] / gap);
        int lower_z_idx = int(mModel.mLowerBound[2] / gap);
        int upper_x_idx = int(mModel.mUpperBound[0] / gap) + 1;
        int upper_y_idx = int(mModel.mUpperBound[1] / gap) + 1;
        int upper_z_idx = int(mModel.mUpperBound[2] / gap) + 1;

        // Put random particles into every grids between the lower and upper grid.
        MeshObject* mMOBJ = construct_mesh_object(mModel.mQmVertData.size() / 3,
                                                  mModel.mQmVertData.data(),
                                                  mModel.mQMIndData.size() / 3,
                                                  mModel.mQMIndData.data());
        glm::mat4 modelInv = glm::inverse(mModel.mModelMat);
        int occ_blocks_num = 0;
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
                            ++occ_blocks_num;
                            tmpParticleGroup.particlePosVec.push_back(node_x);
                            tmpParticleGroup.particlePosVec.push_back(node_y);
                            tmpParticleGroup.particlePosVec.push_back(node_z);
                        }
                    }
                }
            }
        }

        tmpParticleGroup.particleNum = tmpParticleGroup.particlePosVec.size() / 3;
        tmpParticleGroup.mParticleVolume = occ_blocks_num * (gap * gap * gap) / tmpParticleGroup.particleNum;
        tmpParticleGroup.mParticleMass = curInfo.mMaterial.mDensity * tmpParticleGroup.mParticleVolume;

        destroy_mesh_object(mMOBJ);
        initParticles(tmpParticleGroup, curInfo);
        min_bound_x = std::min(min_bound_x, double(mModel.mLowerBound[0]));
        min_bound_y = std::min(min_bound_y, double(mModel.mLowerBound[1]));
        min_bound_z = std::min(min_bound_z, double(mModel.mLowerBound[2]));
        max_bound_x = std::max(max_bound_x, double(mModel.mUpperBound[0]));
        max_bound_y = std::max(max_bound_y, double(mModel.mUpperBound[1]));
        max_bound_z = std::max(max_bound_z, double(mModel.mUpperBound[2]));

        mParticlesGroupsVec.push_back(tmpParticleGroup);
    }

    min_bound_x = min_bound_x - DBL_EPSILON;
    min_bound_y = min_bound_y - DBL_EPSILON;
    min_bound_z = min_bound_z - DBL_EPSILON;
    max_bound_x = max_bound_x + DBL_EPSILON;
    max_bound_y = max_bound_y + DBL_EPSILON;
    max_bound_z = max_bound_z + DBL_EPSILON;
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

    // Free GRAM associated with Particles.
    for (int i = 0; i < mParticlesGroupsVec.size(); ++i) {
        err = cudaFree(mParticlesGroupsVec[i].pPosVecGRAM);
        if (err != cudaSuccess){
            std::cerr << "Free particle position error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        err = cudaFree(mParticlesGroupsVec[i].pVelVecGRAM);
        if (err != cudaSuccess){
            std::cerr << "Free particle velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        err = cudaFree(mParticlesGroupsVec[i].pAffineVelGRAM);
        if (err != cudaSuccess){
            std::cerr << "Free particle affine velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        err = cudaFree(mParticlesGroupsVec[i].pDeformationGradientDiffGRAM);
        if (err != cudaSuccess){
            std::cerr << "Free particle deformation gradient difference error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        err = cudaFree(mParticlesGroupsVec[i].pElasiticityDeformationGradientGRAM);
        if (err != cudaSuccess){
            std::cerr << "Free particle elasticity deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        if (mParticlesGroupsVec[i].mMaterial.mType == SNOW){
            err = cudaFree(mParticlesGroupsVec[i].pPlasiticityDeformationGradientGRAM);
            if (err != cudaSuccess){
                std::cerr << "Free particle plasticity deformation gradient error." << std::endl << cudaGetErrorString(err) << std::endl;
                exit(1);
            }
        }else if (mParticlesGroupsVec[i].mMaterial.mType == WATER){
            err = cudaFree(mParticlesGroupsVec[i].pJVecGRAM);
            if (err != cudaSuccess){
                std::cerr << "Free water particle J vector error." << std::endl << cudaGetErrorString(err) << std::endl;
                exit(1);
            }
        }

    }

}

int MPMSimulator::totalParticlesNum() {
    int res = 0;
    for (int i = 0; i < mParticlesGroupsVec.size(); ++i) {
        res += mParticlesGroupsVec[i].particleNum;
    }
    return res;
}

void MPMSimulator::getGLParticlesPos(std::vector<float> &oPosVec) {
    oPosVec.resize(totalParticlesNum() * 3);
    int already_copy_num = 0;
    for (int i = 0; i < mParticlesGroupsVec.size(); ++i) {
        std::copy(mParticlesGroupsVec[i].particlePosVec.begin(),
                  mParticlesGroupsVec[i].particlePosVec.end(),
                  oPosVec.begin() + already_copy_num);
        already_copy_num += (mParticlesGroupsVec[i].particleNum * 3);
    }
}


