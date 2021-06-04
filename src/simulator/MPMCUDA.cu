//
// Created by jiaruiyan on 5/28/21.
//
//

#include "MPMSimulator.cuh"
#include <math.h>
#include <assert.h>
#include <thrust/device_vector.h>
#include "../../thirdparties/cudaSVD/svd3_cuda.h"

__device__ void FixedCorotatedPStressSigma(float sigma1, float sigma2, float sigma3,
                                           float mu, float lambda,
                                           float& dig1, float& dig2, float& dig3){
    dig1 = 2.f * mu * (sigma1 - 1.f) + lambda * (sigma1 * sigma2 * sigma3 - 1.f) * sigma2 * sigma3;
    dig2 = 2.f * mu * (sigma2 - 1.f) + lambda * (sigma1 * sigma2 * sigma3 - 1.f) * sigma1 * sigma3;
    dig3 = 2.f * mu * (sigma3 - 1.f) + lambda * (sigma1 * sigma2 * sigma3 - 1.f) * sigma1 * sigma2;
}

__device__ void FixedCorotatedPStress(float F11, float F12, float F13,
                                      float F21, float F22, float F23,
                                      float F31, float F32, float F33,
                                      float mu, float lambda,
                                      float &P11, float &P12, float &P13,
                                      float &P21, float &P22, float &P23,
                                      float &P31, float &P32, float &P33){
    float U11, U12, U13, U21, U22, U23, U31, U32, U33;
    float V11, V12, V13, V21, V22, V23, V31, V32, V33;
    float S11, S22, S33;
    float dig1, dig2, dig3;

    svd(F11, F12, F13, F21, F22, F23, F31, F32, F33,
        U11, U12, U13, U21, U22, U23, U31, U32, U33,
        S11, S22, S33,
        V11, V12, V13, V21, V22, V23, V31, V32, V33);
    FixedCorotatedPStressSigma(S11, S22, S33, mu, lambda, dig1, dig2, dig3);
}

/*
__device__ void MatTranspose(const double* m, double* res, const int mRowNum, const int mColNum){
    for (int mRowIdx = 0; mRowIdx < mRowNum; ++mRowIdx){
        for (int mColIdx = 0; mColIdx < mColNum; ++mColIdx){
            int mIdx = mRowIdx * mColNum + mColIdx;
            int resIdx = mColIdx * mRowNum + mRowIdx;
            res[resIdx] = m[mIdx];
        }
    }
}
*/

template<class T>
__forceinline__
__device__ void MatTranspose(const T* x, T* transpose) {
    transpose[0]=x[0]; transpose[1]=x[3]; transpose[2]=x[6];
    transpose[3]=x[1]; transpose[4]=x[4]; transpose[5]=x[7];
    transpose[6]=x[2]; transpose[7]=x[5]; transpose[8]=x[8];
}

template<class T>
__device__ void MatAdd(const T* m1, const T* m2, T* mAdd, int eleNum){
    for (int i = 0; i < eleNum; ++i){
        mAdd[i] = m1[i] + m2[i];
    }
}

/*
__device__ void MatMul(const double* m1, const double* m2, double* mMul,
                       int m1RowNum, int m1ColNum, int m2RowNum, int m2ColNum){
    assert(m1ColNum == m2RowNum);
    // The dot product between the m1_row_idx of the m1 row and the m2_col_idx of the m2 col.
    for (int m1_row_idx = 0; m1_row_idx < m1RowNum; ++m1_row_idx){
        for (int m2_col_idx = 0; m2_col_idx < m2ColNum; ++m2_col_idx){
            double res = 0.0;
            for (int m1_col_idx = 0; m1_col_idx < m1ColNum; ++m1_col_idx){
                int m1_idx = m1_row_idx * m1ColNum + m1_col_idx;
                int m2_idx = m1_col_idx * m2ColNum + m2_col_idx;
                res += (m1[m1_idx] * m2[m2_idx]);
            }
            int mIdx = m1_row_idx * m2ColNum + m2_col_idx;
            mMul[mIdx] = res;
        }
    }
}
*/

template<class T>
__forceinline__
__device__ void MatMul(const T* a, const T* b, T* c)
{
    c[0]=a[0]*b[0]+a[3]*b[1]+a[6]*b[2];
    c[1]=a[1]*b[0]+a[4]*b[1]+a[7]*b[2];
    c[2]=a[2]*b[0]+a[5]*b[1]+a[8]*b[2];
    c[3]=a[0]*b[3]+a[3]*b[4]+a[6]*b[5];
    c[4]=a[1]*b[3]+a[4]*b[4]+a[7]*b[5];
    c[5]=a[2]*b[3]+a[5]*b[4]+a[8]*b[5];
    c[6]=a[0]*b[6]+a[3]*b[7]+a[6]*b[8];
    c[7]=a[1]*b[6]+a[4]*b[7]+a[7]*b[8];
    c[8]=a[2]*b[6]+a[5]*b[7]+a[8]*b[8];
}

template<class T>
__forceinline__
__device__ __host__ void matrixVectorMultiplication(const T* x, const T* v, T* result)
{
    result[0]=x[0]*v[0]+x[3]*v[1]+x[6]*v[2];
    result[1]=x[1]*v[0]+x[4]*v[1]+x[7]*v[2];
    result[2]=x[2]*v[0]+x[5]*v[1]+x[8]*v[2];
}

template<class T>
__device__ void ScalarMatMul(const T scalar, const T* mat, T* res, int matEleNum){
    for (int i = 0; i < matEleNum; ++i){
        res[i] = scalar * mat[i];
    }
}

__device__ void OuterProduct(const double* v1, const double* v2, double* res, int vecEleNum){
    for (int i = 0; i < vecEleNum; ++i){
        for (int j = 0; j < vecEleNum; ++j){
            int res_idx = i * vecEleNum + j;
            res[res_idx] = v1[i] * v2[i];
        }
    }
}

__device__ double BSplineInterpolation1DDerivative(const double x){
    if (x > -0.5 && x < 0.5){
        return -2.0 * x;
    }else if (x >= 0.5 && x < 1.5){
        return x - 1.5;
    }else if (x > -1.5 && x <= -0.5){
        return 1.5 + x;
    }else{
        return 0.0;
    }
}

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

__device__ void BSplineInterpolationGradient(const double xp[3], const double xi[3], const double h,
                                             double& gx, double& gy, double&gz){
    double h_inv = 1.0 / h;
    double i1 = h_inv * (xp[0] - xi[0]);
    double i2 = h_inv * (xp[1] - xi[1]);
    double i3 = h_inv * (xp[2] - xi[2]);
    gx = h_inv * BSplineInterpolation1DDerivative(i1) * BSplineInterpolation1D(i2) * BSplineInterpolation1D(i3);
    gy = h_inv * BSplineInterpolation1D(i1) * BSplineInterpolation1DDerivative(i2) * BSplineInterpolation1D(i3);
    gz = h_inv * BSplineInterpolation1D(i1) * BSplineInterpolation1D(i2) * BSplineInterpolation1DDerivative(i3);
}

__device__ double BSplineInterpolation(const double xp[3], const double xi[3], const double h){
    // printf("Interpolation:(%f, %f, %f)\n", (xp[0] - xi[0]) / h, (xp[1] - xi[1]) / h, (xp[2] - xi[2]) / h);
    return BSplineInterpolation1D((xp[0] - xi[0]) / h) *
           BSplineInterpolation1D((xp[1] - xi[1]) / h) *
           BSplineInterpolation1D((xp[2] - xi[2]) / h);

}

__global__ void P2G(unsigned int pNum,
                    double* pPosVec, double* pMassVec, double* pVelVec, double* pDGVec, double* pVolVec,
                    double gOriCorner_x, double gOriCorner_y, double gOriCorner_z,
                    unsigned int gNodeNumDim, double h, double mu, double lambda,
                    double* gNodeMassVec, double* gNodeTmpMotVec, double* gElasticityForceVec){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < pNum){
        double pos[3] = {pPosVec[i * 3], pPosVec[i * 3 + 1], pPosVec[i * 3 + 2]};
        double m = pMassVec[i];
        double vel[3] = {pVelVec[i * 3], pVelVec[i * 3 + 1], pVelVec[i * 3 + 2]};
        float tmpDeformationGradient[9] = {float(pDGVec[9 * i]), float(pDGVec[9 * i + 1]), float(pDGVec[9 * i + 2]),
                                           float(pDGVec[9 * i + 3]), float(pDGVec[9 * i + 4]), float(pDGVec[9 * i + 5]),
                                           float(pDGVec[9 * i + 6]), float(pDGVec[9 * i + 7]), float(pDGVec[9 * i + 8])};
        float stress[9] = {0.0};
        FixedCorotatedPStress(tmpDeformationGradient[0], tmpDeformationGradient[1], tmpDeformationGradient[2],
                              tmpDeformationGradient[3], tmpDeformationGradient[4], tmpDeformationGradient[5],
                              tmpDeformationGradient[6], tmpDeformationGradient[7], tmpDeformationGradient[8],
                              float(mu), float(lambda),
                              stress[0], stress[1], stress[2],
                              stress[3], stress[4], stress[5],
                              stress[6], stress[7], stress[8]);
        float F_transpose[9] = {0.f};
        MatTranspose(tmpDeformationGradient, F_transpose);
        float tmpMat[9] = {0.f};
        MatMul(stress, F_transpose, tmpMat);
        float mVol(pVolVec[i]);
        ScalarMatMul(mVol, tmpMat, tmpMat, 9);

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

                    // Transfer elasticity force to grid.
                    double grad_wip[3] = {0.0};
                    float tmpForce[3] = {0.f};
                    BSplineInterpolationGradient(pos, b_pos, h, grad_wip[0], grad_wip[1], grad_wip[2]);
                    float grad_wip_f[3] = {static_cast<float>(grad_wip[0]), static_cast<float>(grad_wip[1]), static_cast<float>(grad_wip[2])};
                    matrixVectorMultiplication(tmpMat, grad_wip_f, tmpForce);
                    atomicAdd(&gElasticityForceVec[3 * g_idx], -tmpForce[0]);
                    atomicAdd(&gElasticityForceVec[3 * g_idx + 1], -tmpForce[1]);
                    atomicAdd(&gElasticityForceVec[3 * g_idx + 2], -tmpForce[2]);
                }
            }
        }
        assert(abs(t_w - 1.0) < 0.001);
        assert(abs(t_m - 1.0) < 0.001);
    }
}

__global__ void VelUpdate(unsigned int gNum, double dt, double ext_gravity,
                          double* gMassVec, double* gVelMotVec, double* gForceVec){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < gNum){
        double mass = gMassVec[i];
        if (abs(mass) < 0.00001){
            gMassVec[i] = 0.0;
            gVelMotVec[3 * i] = 0.0;
            gVelMotVec[3 * i + 1] = 0.0;
            gVelMotVec[3 * i + 2] = 0.0;
            gForceVec[3 * i] = 0.0;
            gForceVec[3 * i + 1] = 0.0;
            gForceVec[3 * i + 2] = 0.0;
        }
        else{
            // Calculate velocity from momentum.
            gVelMotVec[3 * i] = gVelMotVec[3 * i] / mass;
            gVelMotVec[3 * i + 1] = gVelMotVec[3 * i + 1] / mass;
            gVelMotVec[3 * i + 2] = gVelMotVec[3 * i + 2] / mass;
            // Include gravity into velocity.
            gVelMotVec[3 * i + 1] = gVelMotVec[3 * i + 1] + ext_gravity * dt;
            // Include elasticity force into velocity.
            gVelMotVec[3 * i] += (dt * gForceVec[3 * i] / mass);
            gVelMotVec[3 * i + 1] += (dt * gForceVec[3 * i + 1] / mass);
            gVelMotVec[3 * i + 2] += (dt * gForceVec[3 * i + 2] / mass);
        }
    }
}

__global__ void InterpolateAndMove(unsigned int pNum, double dt,
                                   double* pPosVec, double* pVelVec, double* pDGVec,
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
        double vel_p[3] = {pVelVec[3 * i], pVelVec[3 * i + 1], pVelVec[3 * i + 2]};

        // Update deformation gradient
        double grad_v[9] = {0.0};
        for (int idx_x_offset = 0; idx_x_offset < 3; ++idx_x_offset){
            for (int idx_y_offset = 0; idx_y_offset < 3; ++idx_y_offset){
                for (int idx_z_offset = 0; idx_z_offset < 3; ++idx_z_offset){
                    int idx_x = b_idx_x + idx_x_offset;
                    int idx_y = b_idx_y + idx_y_offset;
                    int idx_z = b_idx_z + idx_z_offset;
                    double b_pos[3] = {gOriCorner_x + idx_x * h,
                                       gOriCorner_y + idx_y * h,
                                       gOriCorner_z + idx_z * h};
                    double grad_vec[3] = {0.0};
                    BSplineInterpolationGradient(pos, b_pos, h, grad_vec[0], grad_vec[1], grad_vec[2]);

                    double add_mat[9] = {0.0};
                    double tmp_grad_v[9];
                    memcpy(tmp_grad_v, grad_v, sizeof(double) * 9);
                    MatAdd(tmp_grad_v, add_mat, grad_v, 9);
                }
            }
        }
        double Fp[9] = {pDGVec[9 * i], pDGVec[9 * i + 1], pDGVec[9 * i + 2],
                        pDGVec[9 * i + 3], pDGVec[9 * i + 4], pDGVec[9 * i + 5],
                        pDGVec[9 * i + 6], pDGVec[9 * i + 7], pDGVec[9 * i + 8]};
        double leftMat[9] = {1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0};
        double tmp_leftMat[9] = {1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0};
        ScalarMatMul(dt, grad_v, grad_v, 9);
        MatAdd(tmp_leftMat, grad_v, leftMat, 9);
        double tmp_Fp[9];
        memcpy(tmp_Fp, Fp, sizeof(double) * 9);
        MatMul(leftMat, tmp_Fp, Fp);
        pDGVec[9 * i] = Fp[0];
        pDGVec[9 * i + 1] = Fp[1];
        pDGVec[9 * i + 2] = Fp[2];
        pDGVec[9 * i + 3] = Fp[3];
        pDGVec[9 * i + 4] = Fp[4];
        pDGVec[9 * i + 5] = Fp[5];
        pDGVec[9 * i + 6] = Fp[6];
        pDGVec[9 * i + 7] = Fp[7];
        pDGVec[9 * i + 8] = Fp[8];

        /*
        if (i == 0){
            printf("Deformation Gradient of particle 0:\n");
            printf("[%f, %f, %f]\n[%f, %f, %f]\n[%f, %f, %f]\n", Fp[0], Fp[1], Fp[2], Fp[3], Fp[4], Fp[5], Fp[6], Fp[7], Fp[8]);
        }
        */

        // Apply velocity
        pPosVec[3 * i] += dt * vel_p[0];
        pPosVec[3 * i + 1] += dt * vel_p[1];
        pPosVec[3 * i + 2] += dt * vel_p[2];
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
    err = cudaMemset(mGrid.nodeForceVec, 0, mGrid.forceVecByteSize);
    if (err != cudaSuccess){
        std::cerr << "Clean grid force error." << std::endl << cudaGetErrorString(err) << std::endl;
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
                                            mParticles.pDeformationGradientGRAM,
                                            mParticles.pVolVecGRAM,
                                            mGrid.originCorner[0], mGrid.originCorner[1], mGrid.originCorner[2],
                                            mGrid.nodeNumDim,
                                            mGrid.h, mParticles.mMaterialVec[0].mMu, mParticles.mMaterialVec[0].mLambda,
                                            mGrid.nodeMassVec,
                                            mGrid.nodeVelVec,
                                            mGrid.nodeForceVec);
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
    VelUpdate<<<gBlocksPerGrid, gThreadsPerBlock>>>(gNum, dt, ext_gravity,
                                                    mGrid.nodeMassVec, mGrid.nodeVelVec, mGrid.nodeForceVec);
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
                                                             mParticles.pDeformationGradientGRAM,
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


