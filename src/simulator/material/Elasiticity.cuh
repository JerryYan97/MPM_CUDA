//
// Created by jiaruiyan on 6/2/21.
//

#ifndef JIARUI_MPM_ELASITICITY_CUH
#define JIARUI_MPM_ELASITICITY_CUH

#include <iostream>

enum MaterialType{
    JELLO, SNOW, WATER, SAND
};

class Material{
private:
    double mYoungsModulus;
    double mPoissonRatio;

public:
    double mLambda;
    double mMu;
    double mDensity;

    MaterialType mType;

    Material():
    mYoungsModulus(0.0), mPoissonRatio(0.0), mLambda(0.0), mMu(0.0), mDensity(0.0), mType(JELLO)
    {}

    Material(double iYM, double iPR, double iDensity, MaterialType iType):
    mType(iType){
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
