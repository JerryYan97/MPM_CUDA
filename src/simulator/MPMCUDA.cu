//
// Created by jiaruiyan on 5/28/21.
//
//

#include "MPMSimulator.cuh"
#include <math.h>
#include <assert.h>
#include <thrust/device_vector.h>


__device__ double BSplineInterpolation1D(const double x){
    double abs_x = abs(x);
    if (abs_x >= 0 && abs_x < 0.5){
        return 0.75 - abs_x * abs_x;
    }
    else if (abs_x >= 0.5 && abs_x < 1.5){
        return 0.5 * (1.5 - abs_x) * (1.5 - abs_x);
    }
    else{
        return 0.0;
    }
}

__device__ double BSplineInterpolation(const double xp[3], const double xi[3], const double h){
    // printf("Interpolation:(%f, %f, %f)\n", (xp[0] - xi[0]) / h, (xp[1] - xi[1]) / h, (xp[2] - xi[2]) / h);
    return BSplineInterpolation1D((xp[0] - xi[0]) / h) *
           BSplineInterpolation1D((xp[1] - xi[1]) / h) *
           BSplineInterpolation1D((xp[2] - xi[2]) / h);

}

__global__ void P2G(unsigned int pNum,
                    double* pPosVec, double* pMassVec, double* pVelVec,
                    double gOriCorner_x, double gOriCorner_y, double gOriCorner_z,
                    unsigned int gNodeNumDim, double h,
                    double* gNodeMassVec, double* gNodeTmpMotVec){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < pNum){
        double pos[3] = {pPosVec[i * 3], pPosVec[i * 3 + 1], pPosVec[i * 3 + 2]};
        double m = pMassVec[i];
        double vel[3] = {pVelVec[i * 3], pVelVec[i * 3 + 1], pVelVec[i * 3 + 2]};
        int b_idx_x = max(0, int((pos[0] - gOriCorner_x - 0.5 * h) / h));
        int b_idx_y = max(0, int((pos[1] - gOriCorner_y - 0.5 * h) / h));
        int b_idx_z = max(0, int((pos[2] - gOriCorner_z - 0.5 * h) / h));
        double t_w = 0.0;
        double t_m = 0.0;
        for (int idx_x_offset = 0; idx_x_offset < 3; ++idx_x_offset){
            for (int idx_y_offset = 0; idx_y_offset < 3; ++idx_y_offset){
                for (int idx_z_offset = 0; idx_z_offset < 3; ++idx_z_offset){
                    int idx_x = b_idx_x + idx_x_offset;
                    int idx_y = b_idx_y + idx_y_offset;
                    int idx_z = b_idx_z + idx_z_offset;
                    double b_pos[3] = {gOriCorner_x + idx_x * h,
                                       gOriCorner_y + idx_y * h,
                                       gOriCorner_z + idx_z * h};
                    // printf("b_pos:(%f, %f, %f)\n", b_pos[0], b_pos[1], b_pos[2]);
                    double w = BSplineInterpolation(pos, b_pos, h);
                    int g_idx = idx_z * gNodeNumDim * gNodeNumDim + idx_y * gNodeNumDim + idx_x;
                    assert(g_idx < gNodeNumDim * gNodeNumDim * gNodeNumDim);
                    assert(g_idx >= 0);

                    atomicAdd(&gNodeMassVec[g_idx], w * m);
                    t_m += w * m;

                    atomicAdd(&gNodeTmpMotVec[3 * g_idx], w * m * vel[0]);
                    atomicAdd(&gNodeTmpMotVec[3 * g_idx + 1], w * m * vel[1]);
                    atomicAdd(&gNodeTmpMotVec[3 * g_idx + 2], w * m * vel[2]);
                    t_w += w;
                }
            }
        }
        assert(abs(t_w - 1.0) < 0.001);
        assert(abs(t_m - 1.0) < 0.001);
    }
}

__global__ void VelUpdate(unsigned int gNum, double dt, double* gMassVec, double* gVelMotVec){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < gNum){
        double mass = gMassVec[i];
        if (abs(mass) < 0.00001){
            gMassVec[i] = 0.0;
            gVelMotVec[3 * i] = 0.0;
            gVelMotVec[3 * i + 1] = 0.0;
            gVelMotVec[3 * i + 2] = 0.0;
        }
        else{
            // Calculate velocity from momentum.
            gVelMotVec[3 * i] = gVelMotVec[3 * i] / mass;
            gVelMotVec[3 * i + 1] = gVelMotVec[3 * i + 1] / mass;
            gVelMotVec[3 * i + 2] = gVelMotVec[3 * i + 2] / mass;
            // Include gravity into velocity.
            gVelMotVec[3 * i + 1] = gVelMotVec[3 * i + 1] - 9.8 * dt;
            // printf("gNode vel:[%f, %f, %f]", gVelMotVec[3 * i], gVelMotVec[3 * i + 1], gVelMotVec[3 * i + 2]);
        }
    }
}

__global__ void InterpolateAndMove(unsigned int pNum, double dt,
                                   double* pPosVec, double* pVelVec,
                                   double gOriCorner_x, double gOriCorner_y, double gOriCorner_z,
                                   unsigned int gNodeNumDim, double h, double* gNodeVelVec){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < pNum){
        double pos[3] = {pPosVec[i * 3], pPosVec[i * 3 + 1], pPosVec[i * 3 + 2]};
        int b_idx_x = max(0, int((pos[0] - gOriCorner_x - 0.5 * h) / h));
        int b_idx_y = max(0, int((pos[1] - gOriCorner_y - 0.5 * h) / h));
        int b_idx_z = max(0, int((pos[2] - gOriCorner_z - 0.5 * h) / h));
        double t_w = 0.0;
        double t_vel_x = 0.0;
        double t_vel_y = 0.0;
        double t_vel_z = 0.0;
        for (int idx_x_offset = 0; idx_x_offset < 3; ++idx_x_offset){
            for (int idx_y_offset = 0; idx_y_offset < 3; ++idx_y_offset){
                for (int idx_z_offset = 0; idx_z_offset < 3; ++idx_z_offset){
                    int idx_x = b_idx_x + idx_x_offset;
                    int idx_y = b_idx_y + idx_y_offset;
                    int idx_z = b_idx_z + idx_z_offset;
                    double b_pos[3] = {gOriCorner_x + idx_x * h,
                                       gOriCorner_y + idx_y * h,
                                       gOriCorner_z + idx_z * h};

                    double w = BSplineInterpolation(pos, b_pos, h);
                    int g_idx = idx_z * gNodeNumDim * gNodeNumDim + idx_y * gNodeNumDim + idx_x;
                    assert(g_idx < gNodeNumDim * gNodeNumDim * gNodeNumDim);
                    assert(g_idx >= 0);

                    pVelVec[3 * i] += w * gNodeVelVec[3 * g_idx];
                    pVelVec[3 * i + 1] += w * gNodeVelVec[3 * g_idx + 1];
                    pVelVec[3 * i + 2] += w * gNodeVelVec[3 * g_idx + 2];

                    t_vel_x += gNodeVelVec[3 * g_idx];
                    t_vel_y += gNodeVelVec[3 * g_idx + 1];
                    t_vel_z += gNodeVelVec[3 * g_idx + 2];
                    // printf("gNode vel:[%f, %f, %f]\n", gNodeVelVec[3 * g_idx], gNodeVelVec[3 * g_idx + 1], gNodeVelVec[3 * g_idx + 2]);
                    t_w += w;
                }
            }
        }
        assert(abs(t_w - 1.0) < 0.0001);
        // Apply velocity
        pPosVec[3 * i] += dt * pVelVec[3 * i];
        pPosVec[3 * i + 1] += dt * pVelVec[3 * i + 1];
        pPosVec[3 * i + 2] += dt * pVelVec[3 * i + 2];
        // printf("total vel:[%f, %f, %f]\n", t_vel_x, t_vel_y, t_vel_z);
        // printf("pVel:[%f, %f, %f]\n", t_vel_x, t_vel_y, t_vel_z);
    }
}

void MPMSimulator::step() {

    // 0. Check each particles is affected by 9 grid nodes and within the grid.
    cudaError_t err = cudaSuccess;

    // 1. Clean grid data.
    err = cudaMemset(mGrid.nodeMassVec, 0, mGrid.massVecByteSize);
    if(err != cudaSuccess){
        std::cerr << "Clean grid mass error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemset(mGrid.nodeVelVec, 0, mGrid.velVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Clean grid velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // 2. Transfer mass to the grid.
    // 3. Transfer velocity(Momentum) to the grid.
    int pThreadsPerBlock = 256;
    int pBlocksPerGrid = (mParticles.particleNum + pThreadsPerBlock - 1) / pThreadsPerBlock;
    P2G<<<pBlocksPerGrid, pThreadsPerBlock>>>(mParticles.particleNum,
                                            mParticles.pPosVecGRAM,
                                            mParticles.pMassVecGRAM,
                                            mParticles.pVelVecGRAM,
                                            mGrid.originCorner[0], mGrid.originCorner[1], mGrid.originCorner[2],
                                            mGrid.nodeNumDim,
                                            mGrid.h,
                                            mGrid.nodeMassVec,
                                            mGrid.nodeVelVec);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch P2G kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    double pos0_y_init = mParticles.particlePosVec[1];
    double vel0_y_init = mParticles.particleVelVec[1];
    double vel_energy_init = 0.5 * 1.0 * vel0_y_init * vel0_y_init;

    // Check whether grid mass is equal to particles mass.
    double* h_gMassVec = (double*)malloc(mGrid.massVecByteSize);
    err = cudaMemcpy(h_gMassVec, mGrid.nodeMassVec, mGrid.massVecByteSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cerr << "Copy grid mass memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    thrust::device_vector<double> gMass(h_gMassVec, h_gMassVec + mGrid.massVecByteSize / sizeof(double));
    thrust::device_vector<double> pMass(mParticles.particleMassVec.begin(), mParticles.particleMassVec.end());

    double gSum = thrust::reduce(gMass.begin(), gMass.end());
    double pSum = thrust::reduce(pMass.begin(), pMass.end());
    if (abs(gSum - pSum) > 0.0001){
        std::cerr << "Mass is different between Grid and particles after P2G." << std::endl;
        std::cerr << "gSum:" << gSum << " pSum:" << pSum << std::endl;
        exit(1);
    }
    free(h_gMassVec);
#endif

    // 4. Calculate the velocity.
    // 5. Apply gravity.
    int gNum = mGrid.nodeNumDim * mGrid.nodeNumDim * mGrid.nodeNumDim;
    int gThreadsPerBlock = 256;
    int gBlocksPerGrid = (gNum + gThreadsPerBlock - 1) / gThreadsPerBlock;
    VelUpdate<<<gBlocksPerGrid, gThreadsPerBlock>>>(gNum, dt, mGrid.nodeMassVec, mGrid.nodeVelVec);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch VelUpdate kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 5.5 Clean the velocity on the particles.
    // 6. Interpolate new velocity back to particles.
    // 7. Move particles.
    err = cudaMemset(mParticles.pVelVecGRAM, 0, mParticles.velVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Clean grid velocity error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    std::fill(mParticles.particleVelVec.begin(), mParticles.particleVelVec.end(), 0.0);
    InterpolateAndMove<<<pBlocksPerGrid, pThreadsPerBlock>>>(mParticles.particleNum,
                                                             dt,
                                                             mParticles.pPosVecGRAM,
                                                             mParticles.pVelVecGRAM,
                                                             mGrid.originCorner[0],
                                                             mGrid.originCorner[1],
                                                             mGrid.originCorner[2],
                                                             mGrid.nodeNumDim,
                                                             mGrid.h,
                                                             mGrid.nodeVelVec);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch InterpolateAndMove kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Put particles' position and velocity back to RAM.
    err = cudaMemcpy(mParticles.particlePosVec.data(), mParticles.pPosVecGRAM, mParticles.posVecByteSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cerr << "Copy particle position memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(mParticles.particleVelVec.data(), mParticles.pVelVecGRAM, mParticles.velVecByteSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cerr << "Copy particle velocity memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    t += dt;

#ifdef DEBUG
    // Check whether the energy is consistent.
    std::cout << "vel:[" << mParticles.particleVelVec[0] << " " << mParticles.particleVelVec[1] << " " << mParticles.particleVelVec[2] << "]" << std::endl;
    std::cout << "pos:[" << mParticles.particlePosVec[0] << " " << mParticles.particlePosVec[1] << " " << mParticles.particlePosVec[2] << "]" << std::endl << std::endl;

    double vel_energy_cur = 0.5 * 1.0 * (mParticles.particleVelVec[1] * mParticles.particleVelVec[1]);
    double vel_energy_diff = vel_energy_cur - vel_energy_init;
    double pos0_y_cur = mParticles.particlePosVec[1];
    double gravity_energy = -1.0 * 9.8 * (pos0_y_cur - pos0_y_init);
    std::cout << "vel energy difference:" << vel_energy_diff << " gravity energy difference:" << gravity_energy << std::endl;
#endif
}