//
// Created by jiaruiyan on 5/26/21.
//

#ifndef JIARUI_MPM_MPMSIMULATOR_H
#define JIARUI_MPM_MPMSIMULATOR_H

#include <vector>
#include <array>

// In 3D
struct ParticleGroup{
    unsigned int particleNum;
    std::vector<double> particlePosVec;
    std::vector<double> particleMassVec;
    std::vector<double> particleVelVec;
};

struct Grid{
    double h;
    unsigned int nodeNumDim; // The number of node for each dimension. We assume the grid is a cube.
    std::array<double, 3> originCorner;
    std::vector<double> nodeMassVec;
    std::vector<double> nodeVelVec;
};

class MPMSimulator {
public:
    ParticleGroup mParticles{};
    Grid mGrid{};
    MPMSimulator(double gap,
                 unsigned int nodeNumDim,
                 unsigned int particleNumPerCell,
                 std::string& sampleModelPath);

};


#endif //JIARUI_MPM_MPMSIMULATOR_H
