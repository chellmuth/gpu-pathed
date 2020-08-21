#pragma once

#include <vector>

#include "core/camera.h"
#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/mirror.h"
#include "world.h"
#include "scene_data.h"
#include "primitives/sphere.h"
#include "primitives/triangle.h"
#include "core/vec3.h"

namespace rays {

constexpr float defaultLightPosition = -0.6f;
constexpr int defaultMaxDepth = 3;

class Scene {
public:
    Scene(
        Camera &camera,
        SceneData &sceneData
    ) : m_camera(camera),
        m_sceneData(std::move(sceneData)),
        m_maxDepth(defaultMaxDepth),
        m_nextEventEstimation(true)
    {}

    const Camera &getCamera() const { return m_camera; }
    void setCamera(const Camera &camera) { m_camera = camera; }

    const SceneData &getSceneData() const { return m_sceneData; }

    void setMaterialType(int materialID, MaterialType materialType) {
        const MaterialParams &params = *m_sceneData.materialParams[materialID];
        if (params.getMaterialType() == materialType) { return; }

        std::unique_ptr<MaterialParams> newParams;
        switch (materialType) {
        case MaterialType::Lambertian: {
            newParams = std::make_unique<LambertianParams>(Vec3(0.f), Vec3(0.f));
            break;
        }
        case MaterialType::Mirror: {
            newParams = std::make_unique<MirrorParams>();
            break;
        }
        case MaterialType::Glass: {
            newParams = std::make_unique<GlassParams>(1.4f);
            break;
        }
        case MaterialType::Microfacet: {
            newParams = std::make_unique<MicrofacetParams>(0.1f);
            break;
        }
        }

        m_sceneData.materialParams[materialID] = std::move(newParams);
    }

    Vec3 getColor(int materialID) const {
        return m_sceneData.materialParams[materialID]->getAlbedo();
    }

    Vec3 getEmit(int materialID) const {
        return m_sceneData.materialParams[materialID]->getEmit();
    }

    void setColor(int materialID, Vec3 color) {
        m_sceneData.materialParams[materialID]->setAlbedo(color);
    }

    void setEmit(int materialID, Vec3 color) {
        m_sceneData.materialParams[materialID]->setEmit(color);
    }

    float getIOR(int materialID) const {
        return m_sceneData.materialParams[materialID]->getIOR();
    }

    void setIOR(int materialID, float ior) {
        m_sceneData.materialParams[materialID]->setIOR(ior);
    }

    float getAlpha(int materialID) const {
        return m_sceneData.materialParams[materialID]->getAlpha();
    }

    void setAlpha(int materialID, float alpha) {
        m_sceneData.materialParams[materialID]->setAlpha(alpha);
    }

    MaterialType getMaterialType(int materialID) const {
        return m_sceneData.materialParams[materialID]->getMaterialType();
    }

    int getMaxDepth() const {
        return m_maxDepth;
    }
    void setMaxDepth(int maxDepth) {
        m_maxDepth = maxDepth;
    }

    bool getNextEventEstimation() const {
        return m_nextEventEstimation;
    }
    void setNextEventEstimation(bool nextEventEstimation) {
        m_nextEventEstimation = nextEventEstimation;
    }

private:
    Camera m_camera;
    SceneData m_sceneData;
    int m_maxDepth;
    bool m_nextEventEstimation;
};

}


namespace rays { namespace SceneParameters {

SceneData getSceneData(int index);
Camera getCamera(int index, Resolution resolution);

} }
