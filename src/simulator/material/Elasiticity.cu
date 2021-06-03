//
// Created by jiaruiyan on 6/2/21.
//
#include "Elasiticity.cuh"
#include <iostream>
#include "../../thirdparties/cudaSVD/svd3_cuda.h"

__device__ void FixedCorotatedPStress(float F11, float F12, float F13,
                                      float F21, float F22, float F23,
                                      float F31, float F32, float F33,
                                      float mu, float lambda,
                                      float& P11, float& P12, float& P13,
                                      float& P21, float& P22, float& P23,
                                      float& P31, float& P32, float& P33){
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

__device__ void FixedCorotatedPStressSigma(float sigma1, float sigma2, float sigma3,
                                           float mu, float lambda,
                                           float& dig1, float& dig2, float& dig3){
    dig1 = 2.f * mu * (sigma1 - 1.f) + lambda * (sigma1 * sigma2 * sigma3 - 1.f) * sigma2 * sigma3;
    dig2 = 2.f * mu * (sigma2 - 1.f) + lambda * (sigma1 * sigma2 * sigma3 - 1.f) * sigma1 * sigma3;
    dig3 = 2.f * mu * (sigma3 - 1.f) + lambda * (sigma1 * sigma2 * sigma3 - 1.f) * sigma1 * sigma2;
}

FixedCorotatedMaterial::FixedCorotatedMaterial(double iYM, double iPR) {
    mYoungsModulus = iYM;
    mPoissonRatio = iPR;
    if(mYoungsModulus < 0){
        std::cerr << "Youngs Modulus is out of range." << std::endl;
        exit(1);
    }
    if(mPoissonRatio < 0 || mPoissonRatio > 0.5){
        std::cerr << "Poison Ratio is out of range." << std::endl;
        exit(1);
    }
    mMu = mYoungsModulus / (2 * (1 + mPoissonRatio));
    mLambda = mYoungsModulus * mPoissonRatio / ((1 + mPoissonRatio) * (1 - 2 * mPoissonRatio));
}
