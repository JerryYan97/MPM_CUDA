//
// Created by jiaruiyan on 5/26/21.
//

#ifndef JIARUI_MPM_MPMSIMULATOR_H
#define JIARUI_MPM_MPMSIMULATOR_H

#include <vector>
#include <array>
#include <cuda_runtime.h>
#include <iostream>


// In 3D
struct ParticleGroup{
    unsigned int particleNum;
    std::vector<double> particlePosVec;
    std::vector<double> particleMassVec;
    std::vector<double> particleVelVec;
    double* pPosVecGRAM;
    double* pMassVecGRAM;
    double* pVelVecGRAM;
    size_t posVecByteSize;
    size_t massVecByteSize;
    size_t velVecByteSize;
};

struct Grid{
    double h;
    unsigned int nodeNumDim; // The number of node for each dimension. We assume the grid is a cube.
    std::array<double, 3> originCorner;
    double* nodeMassVec;
    double* nodeVelVec;
    size_t massVecByteSize;
    size_t velVecByteSize;
};

class MPMSimulator {
private:
    double dt;

public:
    double t;
    ParticleGroup mParticles{};
    Grid mGrid{};
    MPMSimulator(double gap,
                 double dt,
                 unsigned int nodeNumDim,
                 unsigned int particleNumPerCell,
                 std::string& sampleModelPath);
    void step();
    ~MPMSimulator();
};


#endif //JIARUI_MPM_MPMSIMULATOR_H
