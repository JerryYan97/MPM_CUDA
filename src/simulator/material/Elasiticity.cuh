//
// Created by jiaruiyan on 6/2/21.
//

#ifndef JIARUI_MPM_ELASITICITY_CUH
#define JIARUI_MPM_ELASITICITY_CUH

__device__ void FixedCorotatedPStress(float F11, float F12, float F13,
                                      float F21, float F22, float F23,
                                      float F31, float F32, float F33,
                                      float mu, float lambda,
                                      float& P11, float& P12, float& P13,
                                      float& P21, float& P22, float& P23,
                                      float& P31, float& P32, float& P33);

__device__ void FixedCorotatedPStressSigma(float sigma1, float sigma2, float sigma3,
                                           float mu, float lambda,
                                           float& dig1, float& dig2, float& dig3);

class FixedCorotatedMaterial{
private:
    double mYoungsModulus;
    double mPoissonRatio;

public:
    double mLambda;
    double mMu;
    FixedCorotatedMaterial(double iYM, double iPR);
};


#endif //JIARUI_MPM_ELASITICITY_CUH
