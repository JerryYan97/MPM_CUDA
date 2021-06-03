//
// Created by jiaruiyan on 5/26/21.
//

#ifndef JIARUI_MPM_INSTANCEMODEL_H
#define JIARUI_MPM_INSTANCEMODEL_H

#include "model.h"

// A bunch of instances represent by a same mesh.
class InstanceModel : public model{
public:
    explicit InstanceModel(std::string &InstanceMeshPath, std::vector<float>& posVec, float modelSize);
    void updateInstanceModel(std::vector<float>& posVec);
    int mInstanceNum;
    ~InstanceModel();
private:
    unsigned int mInstancedArrayBO{};
};


#endif //JIARUI_MPM_INSTANCEMODEL_H
