//
// Created by jiaruiyan on 6/2/21.
//

#ifndef JIARUI_MPM_ELASITICITY_CUH
#define JIARUI_MPM_ELASITICITY_CUH

#include <iostream>

class FixedCorotatedMaterial{
private:
    double mYoungsModulus;
    double mPoissonRatio;

public:
    double mLambda;
    double mMu;
    double mDensity;
    FixedCorotatedMaterial(double iYM, double iPR, double iDensity){
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
        mDensity = iDensity;
    }
};


#endif //JIARUI_MPM_ELASITICITY_CUH
