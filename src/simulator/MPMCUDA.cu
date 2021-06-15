//
// Created by jiaruiyan on 5/28/21.
//
// Put Particle1 deformation gradient into python to check whether the stress1/2 has a problem.

#include "MPMSimulator.cuh"
#include <math.h>
#include <assert.h>
#include <thrust/device_vector.h>
#include "../../thirdparties/cudaSVD/svd3_cuda.h"

template<class T>
__device__ void Mat3x3Cofactor(const T* F, T* res){
    res[0] = F[4] * F[8] - F[5] * F[7];
    res[1] = F[5] * F[6] - F[3] * F[8];
    res[2] = F[3] * F[7] - F[4] * F[6];
    res[3] = F[2] * F[7] - F[1] * F[8];
    res[4] = F[0] * F[8] - F[2] * F[6];
    res[5] = F[1] * F[6] - F[0] * F[7];
    res[6] = F[1] * F[5] - F[2] * F[4];
    res[7] = F[2] * F[3] - F[0] * F[5];
    res[8] = F[0] * F[4] - F[1] * F[3];
}

template<class T>
__forceinline__
__device__ T Mat3x3Determinant(const T* X){
    return X[0] * (X[4] * X[8] - X[5] * X[7]) + X[1] * (X[3] * X[8] - X[5] * X[6]) + X[2] * (X[3] * X[7] - X[4] * X[6]);
}

template<class T>
__forceinline__
__device__ void MatMul3x3(const T* A, const T* B, T* C){
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template<class T>
__forceinline__
__device__ void MatTranspose(const T* x, T* transpose) {
    transpose[0]=x[0]; transpose[1]=x[3]; transpose[2]=x[6];
    transpose[3]=x[1]; transpose[4]=x[4]; transpose[5]=x[7];
    transpose[6]=x[2]; transpose[7]=x[5]; transpose[8]=x[8];
}

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

    /*
    if (abs(F11 - 0.f) < FLT_EPSILON){
        F11 = 0.f;
    }
    if (abs(F12 - 0.f) < FLT_EPSILON){
        F12 = 0.f;
    }
    if (abs(F13 - 0.f) < FLT_EPSILON){
        F13 = 0.f;
    }
    if (abs(F21 - 0.f) < FLT_EPSILON){
        F21 = 0.f;
    }
    if (abs(F22 - 0.f) < FLT_EPSILON){
        F22 = 0.f;
    }
    if (abs(F23 - 0.f) < FLT_EPSILON){
        F23 = 0.f;
    }
    if (abs(F31 - 0.f) < FLT_EPSILON){
        F31 = 0.f;
    }
    if (abs(F32 - 0.f) < FLT_EPSILON){
        F32 = 0.f;
    }
    if (abs(F33 - 0.f) < FLT_EPSILON){
        F33 = 0.f;
    }
    */

    svd(F11, F12, F13, F21, F22, F23, F31, F32, F33,
        U11, U12, U13, U21, U22, U23, U31, U32, U33,
        S11, S22, S33,
        V11, V12, V13, V21, V22, V23, V31, V32, V33);
    FixedCorotatedPStressSigma(S11, S22, S33, mu, lambda, dig1, dig2, dig3);

    float V[9] = {V11, V12, V13,
                  V21, V22, V23,
                  V31, V32, V33};
    float U[9] = {U11, U12, U13,
                  U21, U22, U23,
                  U31, U32, U33};

    /*
    assert(S11 > 0.f || abs(S11) <= 1e-6);
    assert(S22 > 0.f || abs(S22) <= 1e-6);
    assert(S11 > S22 || abs(S11 - S22) <= 1e-6);
    assert(S22 > abs(S33) || abs(S22 - abs(S33)) < 1e-6);
    */

    assert(Mat3x3Determinant(U) >= 0.f);
    assert(Mat3x3Determinant(V) >= 0.f);



    float P_sigma[9] = {dig1, 0.f, 0.f,
                        0.f, dig2, 0.f,
                        0.f, 0.f, dig3};
    float V_transpose[9];
    float tmpMat[9] = {0.f};
    float res[9] = {0.f};
    MatTranspose(V, V_transpose);
    MatMul3x3(P_sigma, V_transpose, tmpMat);
    MatMul3x3(U, tmpMat, res);
    P11 = res[0];
    P12 = res[1];
    P13 = res[2];
    P21 = res[3];
    P22 = res[4];
    P23 = res[5];
    P31 = res[6];
    P32 = res[7];
    P33 = res[8];
}

template<class T>
__device__ void MatAdd(const T* m1, const T* m2, T* mAdd, int eleNum){
    for (int i = 0; i < eleNum; ++i){
        mAdd[i] = m1[i] + m2[i];
    }
}

template<class T>
__forceinline__
__device__ __host__ void MatVelMul3x3x3x1(const T* X, const T* V, T* R)
{
    R[0] = X[0] * V[0] + X[1] * V[1] + X[2] * V[2];
    R[1] = X[3] * V[0] + X[4] * V[1] + X[5] * V[2];
    R[2] = X[6] * V[0] + X[7] * V[1] + X[8] * V[2];
}


template<class T>
__device__ void ScalarMatMul(const T scalar, const T* mat, T* res, int matEleNum){
    for (int i = 0; i < matEleNum; ++i){
        res[i] = scalar * mat[i];
    }
}

template<class T>
__forceinline__
__device__ void OuterProduct(const T* v1, const T* v2, T* res){
    res[0] = v1[0] * v2[0];
    res[1] = v1[0] * v2[1];
    res[2] = v1[0] * v2[2];
    res[3] = v1[1] * v2[0];
    res[4] = v1[1] * v2[1];
    res[5] = v1[1] * v2[2];
    res[6] = v1[2] * v2[0];
    res[7] = v1[2] * v2[1];
    res[8] = v1[2] * v2[2];
}

template<class T>
__device__ void FixedCorotatedStress2(const T* F, const T mu, const T lambda, T* P){
    T F_invT[9] = {0.0};
    Mat3x3Cofactor(F, F_invT);
    T J = Mat3x3Determinant(F);

    float U11, U12, U13, U21, U22, U23, U31, U32, U33;
    float V11, V12, V13, V21, V22, V23, V31, V32, V33;
    float S11, S22, S33;

    svd(F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
        U11, U12, U13, U21, U22, U23, U31, U32, U33,
        S11, S22, S33,
        V11, V12, V13, V21, V22, V23, V31, V32, V33);

    float V[9] = {V11, V12, V13,
                  V21, V22, V23,
                  V31, V32, V33};
    float U[9] = {U11, U12, U13,
                  U21, U22, U23,
                  U31, U32, U33};
    float R[9] = {0.f};
    float term1[9] = {0.f};
    float term2[9] = {0.f};

    float V_transpose[9];
    MatTranspose(V, V_transpose);
    MatMul3x3(U, V_transpose, R);

    float min_R[9] = {0.f};
    ScalarMatMul(-1.f, R, min_R, 9);

    float F_min_R[9] = {0.f};
    MatAdd(F, min_R, F_min_R, 9);
    ScalarMatMul(2.f * mu, F_min_R, term1, 9);

    float JF_invT[9] = {0.f};
    Mat3x3Cofactor(F, JF_invT);
    ScalarMatMul(lambda * (J - 1.f), JF_invT, term2, 9);

    MatAdd(term1, term2, P, 9);
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
                    double* pPosVec, double* pMassVec, double* pVelVec, double* pDGVec,
                    double* pVolVec, double* pForceVec, // int* pAttentionLabel,
                    double gOriCorner_x, double gOriCorner_y, double gOriCorner_z, int* gAttentionIdx,
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
        float stress[9] = {0.f};
        FixedCorotatedPStress(tmpDeformationGradient[0], tmpDeformationGradient[1], tmpDeformationGradient[2],
                              tmpDeformationGradient[3], tmpDeformationGradient[4], tmpDeformationGradient[5],
                              tmpDeformationGradient[6], tmpDeformationGradient[7], tmpDeformationGradient[8],
                              float(mu), float(lambda),
                              stress[0], stress[1], stress[2],
                              stress[3], stress[4], stress[5],
                              stress[6], stress[7], stress[8]);
        float stress2[9] = {0.f};
        FixedCorotatedStress2(tmpDeformationGradient, float(mu), float(lambda), stress2);

        // float tar_stress[9] = {0.0};
        // Times_Rotated_dP_dF_FixedCorotated(float(mu), float(lambda), tmpDeformationGradient, );

        float F_transpose[9] = {0.f};
        MatTranspose(tmpDeformationGradient, F_transpose);
        float tmpMat[9] = {0.f};
        // MatMul3x3(stress, F_transpose, tmpMat);
        MatMul3x3(stress2, F_transpose, tmpMat);

        float mVol(pVolVec[i]);
        ScalarMatMul(mVol, tmpMat, tmpMat, 9);
        /*
        if (i == 1){
            printf("P2G: Tmp Mat of particle 1:\n");
            printf("[%f, %f, %f]\n[%f, %f, %f]\n[%f, %f, %f]\n",
                   tmpMat[0], tmpMat[1], tmpMat[2],
                   tmpMat[3], tmpMat[4], tmpMat[5],
                   tmpMat[6], tmpMat[7], tmpMat[8]);
            printf("P2G: Stress of particle 1:\n");
            printf("[%f, %f, %f]\n[%f, %f, %f]\n[%f, %f, %f]\n",
                   stress[0], stress[1], stress[2],
                   stress[3], stress[4], stress[5],
                   stress[6], stress[7], stress[8]);
            printf("P2G: Stress2 of particle 1:\n");
            printf("[%f, %f, %f]\n[%f, %f, %f]\n[%f, %f, %f]\n",
                   stress2[0], stress2[1], stress2[2],
                   stress2[3], stress2[4], stress2[5],
                   stress2[6], stress2[7], stress2[8]);
            printf("P2G: F1:\n");
            printf("[%f, %f, %f]\n[%f, %f, %f]\n[%f, %f, %f]\n",
                   tmpDeformationGradient[0], tmpDeformationGradient[1], tmpDeformationGradient[2],
                   tmpDeformationGradient[3], tmpDeformationGradient[4], tmpDeformationGradient[5],
                   tmpDeformationGradient[6], tmpDeformationGradient[7], tmpDeformationGradient[8]);
            printf("P2G: F1 determinant:%f\n", Mat3x3Determinant(tmpDeformationGradient));
            printf("mu:%f, lambda:%f\n", mu, lambda);
        }
        */

        int b_idx_x = max(0, int((pos[0] - gOriCorner_x - 0.5 * h) / h));
        int b_idx_y = max(0, int((pos[1] - gOriCorner_y - 0.5 * h) / h));
        int b_idx_z = max(0, int((pos[2] - gOriCorner_z - 0.5 * h) / h));
        double t_w = 0.0;
        double t_m = 0.0;
        float t_f_x = 0.0;
        float t_f_y = 0.0;
        float t_f_z = 0.0;
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

                    if (i == 1){
                        // printf("P109743 related node Mot(g_idx=%d)=[%f, %f, %f] w=%f.\n", g_idx,
                        //        w * m * vel[0], w * m * vel[1], w * m * vel[2], w);
                        gAttentionIdx[idx_x_offset * 9 + idx_y_offset * 3 + idx_z_offset] = g_idx;
                    }

                    // Transfer elasticity force to grid.
                    double grad_wip[3] = {0.0};
                    float tmpForce[3] = {0.f};
                    BSplineInterpolationGradient(pos, b_pos, h, grad_wip[0], grad_wip[1], grad_wip[2]);
                    float grad_wip_f[3] = {static_cast<float>(grad_wip[0]), static_cast<float>(grad_wip[1]), static_cast<float>(grad_wip[2])};
                    MatVelMul3x3x3x1(tmpMat, grad_wip_f, tmpForce);
                    atomicAdd(&gElasticityForceVec[3 * g_idx], -tmpForce[0]);
                    atomicAdd(&gElasticityForceVec[3 * g_idx + 1], -tmpForce[1]);
                    atomicAdd(&gElasticityForceVec[3 * g_idx + 2], -tmpForce[2]);
                    /*
                    if (i == 1){
                        printf("Particle %d contributes force to g_idx = %d:[%f, %f, %f]\n", i, g_idx, -tmpForce[0], -tmpForce[1], -tmpForce[2]);
                        printf("Particle %d contributes grad_wip = %d:[%f, %f, %f]\n", i, g_idx, grad_wip_f[0], grad_wip_f[1], grad_wip_f[2]);
                        // pAttentionLabel[i] = 1;
                    }
                    */

                    t_f_x -= tmpForce[0];
                    t_f_y -= tmpForce[1];
                    t_f_z -= tmpForce[2];

                    /*
                    if (i == 0){
                        printf("grad_wip_f:(%f, %f, %f)\n", grad_wip_f[0], grad_wip_f[1], grad_wip_f[2]);
                        printf("tmpForce:(%f, %f, %f)\n", tmpForce[0], tmpForce[1], tmpForce[2]);
                        printf("total force:(%f, %f, %f)\n", t_f_x, t_f_y, t_f_z);
                    }
                    */
                }
            }
        }
        pForceVec[3 * i] = double(t_f_x);
        pForceVec[3 * i + 1] = double(t_f_y);
        pForceVec[3 * i + 2] = double(t_f_z);
        /*
        if (abs(t_f_x) > 1.0 || abs(t_f_y) > 1.0 || abs(t_f_z) > 1.0){
            printf("Particle %d contributes force:[%f, %f, %f]\n", i, t_f_x, t_f_y, t_f_z);
            // pAttentionLabel[i] = 1;
        }

        if (i == 1){
            printf("P1 Elasticity Force Contribution:(%f, %f, %f)\n", t_f_x, t_f_y, t_f_z);
        }
        */

        assert(abs(t_w - 1.0) < 0.001);
        // assert(abs(t_m - 1.0) < 0.001);
    }
}

__global__ void VelUpdate(unsigned int gNum, double dt, double ext_gravity,
                          double lower_x, double lower_y, double lower_z,
                          double upper_x, double upper_y, double upper_z,
                          double gOriCorner_x, double gOriCorner_y, double gOriCorner_z,
                          unsigned int gNodeNumDim, double h,
                          double* gMassVec, double* gVelMotVec, double* gForceVec){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < gNum){
        double mass = gMassVec[i];
        if (abs(mass) < DBL_EPSILON){
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

            // Deal with Boundary condition.
            int idx_x = i % int(gNodeNumDim);
            int idx_y = ((i - idx_x) / int(gNodeNumDim)) % int(gNodeNumDim);
            int idx_z = ((i - idx_x) / int(gNodeNumDim) - idx_y) / int(gNodeNumDim);
            double grid_node_pos[3] = {gOriCorner_x + idx_x * h,
                                       gOriCorner_y + idx_y * h,
                                       gOriCorner_z + idx_z * h};
            if (grid_node_pos[0] <= lower_x || grid_node_pos[0] >= upper_x ||
                grid_node_pos[1] <= lower_y || grid_node_pos[1] >= upper_y ||
                grid_node_pos[2] <= lower_z || grid_node_pos[2] >= upper_z){
                gVelMotVec[3 * i] = 0.0;
                gVelMotVec[3 * i + 1] = 0.0;
                gVelMotVec[3 * i + 2] = 0.0;
            }
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
                    int g_idx = idx_z * gNodeNumDim * gNodeNumDim + idx_y * gNodeNumDim + idx_x;
                    assert(g_idx < gNodeNumDim * gNodeNumDim * gNodeNumDim);
                    assert(g_idx >= 0);

                    double grad_wip[3] = {0.0};
                    BSplineInterpolationGradient(pos, b_pos, h, grad_wip[0], grad_wip[1], grad_wip[2]);

                    double vi[3] = {gNodeVelVec[3 * g_idx], gNodeVelVec[3 * g_idx + 1], gNodeVelVec[3 * g_idx + 2]};

                    /*
                    if (i == 109743){
                        printf("P109743 related node vel(g_idx=%d)=[%f, %f, %f]:\n", g_idx, vi[0], vi[1], vi[2]);
                    }
                    */

                    double add_mat[9] = {0.0};
                    OuterProduct(vi, grad_wip, add_mat);

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
        MatMul3x3(leftMat, tmp_Fp, Fp);
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
        if (i == 5252144){
            printf("Updated: F5252144:\n");
            printf("[%f, %f, %f]\n[%f, %f, %f]\n[%f, %f, %f]\n",
                   Fp[0], Fp[1], Fp[2],
                   Fp[3], Fp[4], Fp[5],
                   Fp[6], Fp[7], Fp[8]);
            printf("Updated: F5252144 determinant:%f\n", Mat3x3Determinant(Fp));
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

__global__ void FindAllRelatedParticles(unsigned int pNum, double* pPosVec,
                                        double gOriCorner_x, double gOriCorner_y, double gOriCorner_z,
                                        double h, unsigned int gNodeNumDim,
                                        int* gAttentionIdx,
                                        int* pAttentionParticleIdx){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < pNum){
        double pos[3] = {pPosVec[i * 3], pPosVec[i * 3 + 1], pPosVec[i * 3 + 2]};
        int b_idx_x = max(0, int((pos[0] - gOriCorner_x - 0.5 * h) / h));
        int b_idx_y = max(0, int((pos[1] - gOriCorner_y - 0.5 * h) / h));
        int b_idx_z = max(0, int((pos[2] - gOriCorner_z - 0.5 * h) / h));

        for (int idx_x_offset = 0; idx_x_offset < 3; ++idx_x_offset){
            for (int idx_y_offset = 0; idx_y_offset < 3; ++idx_y_offset){
                for (int idx_z_offset = 0; idx_z_offset < 3; ++idx_z_offset){
                    int idx_x = b_idx_x + idx_x_offset;
                    int idx_y = b_idx_y + idx_y_offset;
                    int idx_z = b_idx_z + idx_z_offset;
                    int g_idx = idx_z * gNodeNumDim * gNodeNumDim + idx_y * gNodeNumDim + idx_x;
                    for (int j = 0; j < 27; ++j){
                        if (gAttentionIdx[j] == g_idx){
                            pAttentionParticleIdx[i] = 1.0;
                            return;
                        }
                    }
                }
            }
        }
    }
}

__global__ void FindOutBoundParticles(unsigned int pNum, double* pPosVec,
                                      double minBoundX, double minBoundY, double minBoundZ,
                                      double maxBoundX, double maxBoundY, double maxBoundZ,
                                      int* pAttentionParticleLabel){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < pNum){
        double pos[3] = {pPosVec[i * 3], pPosVec[i * 3 + 1], pPosVec[i * 3 + 2]};
        if (pos[0] < minBoundX || pos[0] > maxBoundX ||
            pos[1] < minBoundY || pos[1] > maxBoundY ||
            pos[2] < minBoundZ || pos[2] > maxBoundZ){
            pAttentionParticleLabel[i] = 1;
        }
        else{
            pAttentionParticleLabel[i] = 0;
        }
    }
}

/* This can be done freely during P2G:
void FindRelatedGrid(int tarIdx, double gOriCorner_x, double gOriCorner_y, double gOriCorner_z,
                     double h, std::vector<int>& gAttentionIdx){

}
*/

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

#ifdef DEBUG
    std::cout << "********* Frame " << current_frame << " starts **********" << std::endl << std::endl;
    double standard_speed = ext_gravity * t;
    // Check particle 1 pos and vel:
    std::cout << "n vel:[" << mParticles.particleVelVec[1 * 3] << " " << mParticles.particleVelVec[1 * 3 + 1] << " " << mParticles.particleVelVec[1 * 3 + 2] << "]" << std::endl;
    std::cout << "n pos:[" << mParticles.particlePosVec[1 * 3] << " " << mParticles.particlePosVec[1 * 3 + 1] << " " << mParticles.particlePosVec[1 * 3 + 2] << "]" << std::endl << std::endl;

#endif
    double* pForceVec;
    err = cudaMalloc(&pForceVec, mParticles.particleNum * sizeof(double) * 3);
    if (err != cudaSuccess){
        std::cerr << "Allocate particle force vector error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    int* gAttentionIdx;
    err = cudaMalloc(&gAttentionIdx, 27 * sizeof(int));
    if (err != cudaSuccess){
        std::cerr << "Allocate attention idx error." << std::endl << cudaGetErrorString(err) << std::endl;
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
                                            pForceVec,
                                            mGrid.originCorner[0], mGrid.originCorner[1], mGrid.originCorner[2],
                                            gAttentionIdx,
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

    // Check force on the grid.
    double* h_gForceVec = (double*)malloc(mGrid.forceVecByteSize);
    err = cudaMemcpy(h_gForceVec, mGrid.nodeForceVec, mGrid.forceVecByteSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cerr << "Copy grid force memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    int* h_gAttentionIdx = (int*)malloc(sizeof(int) * 27);
    err = cudaMemcpy(h_gAttentionIdx, gAttentionIdx, sizeof(int) * 27, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cerr << "Copy attentionIdx memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    for (int i = 0; i < 27; ++i){
        int g_idx = h_gAttentionIdx[i];
        std::cout << "f(g_idx=" << g_idx << ")" << "=[" << h_gForceVec[g_idx * 3] << ", " << h_gForceVec[g_idx * 3 + 1] << ", " << h_gForceVec[g_idx * 3 + 2] << "]" << std::endl;
    }

    int* d_pAttentionLabel;
    err = cudaMalloc(&d_pAttentionLabel, mParticles.particleNum * sizeof(int));
    if (err != cudaSuccess){
        std::cerr << "Allocate pAttentionLabel memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemset(d_pAttentionLabel, 0, mParticles.particleNum * sizeof(int));
    if (err != cudaSuccess){
        std::cerr << "Clean AttentionLabel error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    /*
    FindAllRelatedParticles<<<pBlocksPerGrid, pThreadsPerBlock>>>(mParticles.particleNum,
                                                                  mParticles.pPosVecGRAM,
                                                                  mGrid.originCorner[0],
                                                                  mGrid.originCorner[1],
                                                                  mGrid.originCorner[2],
                                                                  mGrid.h,
                                                                  mGrid.nodeNumDim,
                                                                  gAttentionIdx,
                                                                  pAttentionLabel);
    */

    double* h_pForce = (double*)malloc(sizeof(double) * mParticles.particleNum * 3);
    err = cudaMemcpy(h_pForce, pForceVec, sizeof(double) * mParticles.particleNum * 3, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cerr << "Copy pForceVec memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    /* Attention Nodes:
    for (int i = 0; i < mParticles.particleNum; ++i){
        if (h_pAttentionLabel[i] == 1){
            std::cout << "Related nodes are affected by particle_idx = " << i << std::endl;
            std::cout << "f(p_idx = " << i << ") = " << "[" << h_pForce[i * 3] << ", " << h_pForce[i * 3 + 1] << ", " << h_pForce[i * 3 + 2] << "]" << std::endl;
            h_pAttentionIdx.push_back(i);
        }
    }
    */
#endif

    // 4. Calculate the velocity.
    // 5. Apply gravity.
    int gNum = mGrid.nodeNumDim * mGrid.nodeNumDim * mGrid.nodeNumDim;
    int gThreadsPerBlock = 256;
    int gBlocksPerGrid = (gNum + gThreadsPerBlock - 1) / gThreadsPerBlock;
    VelUpdate<<<gBlocksPerGrid, gThreadsPerBlock>>>(gNum, dt, ext_gravity,
                                                    mGrid.originCorner[0] + 10 * mGrid.h,
                                                    mGrid.originCorner[1] + 10 * mGrid.h,
                                                    mGrid.originCorner[2] + 10 * mGrid.h,
                                                    mGrid.originCorner[0] + mGrid.h * mGrid.nodeNumDim - 10 * mGrid.h,
                                                    mGrid.originCorner[1] + mGrid.h * mGrid.nodeNumDim - 10 * mGrid.h,
                                                    mGrid.originCorner[2] + mGrid.h * mGrid.nodeNumDim - 10 * mGrid.h,
                                                    mGrid.originCorner[0], mGrid.originCorner[1], mGrid.originCorner[2],
                                                    mGrid.nodeNumDim, mGrid.h,
                                                    mGrid.nodeMassVec, mGrid.nodeVelVec, mGrid.nodeForceVec);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch VelUpdate kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG

#endif

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
    std::cout << "n+1 vel:[" << mParticles.particleVelVec[1 * 3] << " " << mParticles.particleVelVec[1 * 3 + 1] << " " << mParticles.particleVelVec[1 * 3 + 2] << "]" << std::endl;
    std::cout << "n+1 pos:[" << mParticles.particlePosVec[1 * 3] << " " << mParticles.particlePosVec[1 * 3 + 1] << " " << mParticles.particlePosVec[1 * 3 + 2] << "]" << std::endl << std::endl;
/*
    double vel_energy_cur = 0.5 * 1.0 * (mParticles.particleVelVec[1] * mParticles.particleVelVec[1]);
    double vel_energy_diff = vel_energy_cur - vel_energy_init;
    double pos0_y_cur = mParticles.particlePosVec[1];
    double gravity_energy = -1.0 * 9.8 * (pos0_y_cur - pos0_y_init);
    std::cout << "vel energy difference:" << vel_energy_diff << " gravity energy difference:" << gravity_energy << std::endl;
*/
    // Check the first particle with velocity problem:
    /*
    for (int i = 0; i < mParticles.particleNum; ++i){
        double vel_x = mParticles.particleVelVec[i * 3];
        double vel_y = mParticles.particleVelVec[i * 3 + 1];
        double vel_z = mParticles.particleVelVec[i * 3 + 2];
        if (abs(vel_x) > 0.001 || abs(vel_z) > 0.001){
            std::cout << "Problem particle id:" << i << std::endl; // 54870
            std::cout << "Problem velocity:[" << vel_x << ", " << vel_y << ", " << vel_z << "]" << std::endl;
        }
    }
    */

    /*
    FindOutBoundParticles<<<pBlocksPerGrid, pThreadsPerBlock>>>(mParticles.particleNum,
                                                                mParticles.pPosVecGRAM,
                                                                min_bound_x, min_bound_y, min_bound_z,
                                                                max_bound_x, max_bound_y, max_bound_z,
                                                                d_pAttentionLabel);
    int* h_pAttentionLabel = (int*)malloc(sizeof(int) * mParticles.particleNum);
    std::vector<int> h_pAttentionIdx;
    err = cudaMemcpy(h_pAttentionLabel, d_pAttentionLabel, sizeof(int) * mParticles.particleNum, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cerr << "Copy pAttentionLabel memory error." << std::endl << cudaGetErrorString(err) << std::endl;
        exit(1);
    }*/

    /* Out of bound Particles -- 109743
    for (int i = 0; i < mParticles.particleNum; ++i){
        if (h_pAttentionLabel[i] == 1){
            std::cout << "Out of bound particle_idx = " << i << std::endl;
            std::cout << "f(p_idx = " << i << ") = " << "[" << h_pForce[i * 3] << ", " << h_pForce[i * 3 + 1] << ", " << h_pForce[i * 3 + 2] << "]" << std::endl;
            std::cout << "v(n+1)(p_idx = " << i << ") = [" << mParticles.particleVelVec[i * 3] << " " << mParticles.particleVelVec[i * 3 + 1] << " " << mParticles.particleVelVec[i * 3 + 2] << "]" << std::endl;
            h_pAttentionIdx.push_back(i);
        }
    }
    */

    std::cout << "***********************************" << std::endl << std::endl;
#endif
    cudaFree(pForceVec);
    cudaFree(gAttentionIdx);
    ++current_frame;
    current_time += dt;
}


