//
// Created by jiaruiyan on 6/13/21.
//

#include "FilesIO.h"
void OutputBego(const std::string& fileNamePath, const std::vector<float>& outPosData){

    int particleNum = outPosData.size() / 3;

    Partio::ParticlesDataMutable* mParticlesData = Partio::create();
    Partio::ParticlesDataMutable::iterator mItr = mParticlesData->addParticles(particleNum);
    mParticlesData->addAttribute("position", Partio::VECTOR, 3);

    Partio::ParticleAttribute mParticlePosAttr;
    mParticlesData->attributeInfo("position", mParticlePosAttr);

    for (int i = 0; i < particleNum; ++i) {
        float tmpPos[3] = {outPosData[i * 3], outPosData[i * 3 + 1], outPosData[i * 3 + 2]};
        mParticlesData->set(mParticlePosAttr, i, tmpPos);
    }

    Partio::write(fileNamePath.c_str(), *mParticlesData);
    mParticlesData->release();
}