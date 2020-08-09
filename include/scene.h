#pragma once

#include <vector>

#include "camera.h"
#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/mirror.h"
#include "primitive.h"
#include "scene_data.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"

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

    void init();
    void update();

    const Camera &getCamera() const { return m_camera; }
    void setCamera(const Camera &camera) { m_camera = camera; }

    const SceneData &getSceneData() const { return m_sceneData; }

    const std::vector<Lambertian> &getLambertians() const {
        return m_sceneData.materialStore.getLambertians();
    }

    const std::vector<Mirror> &getMirrors() const {
        return m_sceneData.materialStore.getMirrors();
    }

    const std::vector<Glass> &getGlasses() const {
        return m_sceneData.materialStore.getGlasses();
    }

    void setMaterialType(int materialID, MaterialType materialType) {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);
        if (materialIndex.materialType == materialType) { return; }

        MaterialStore &store = m_sceneData.materialStore;

        int newID = -1;
        switch (materialType) {
        case MaterialType::Lambertian: {
            Lambertian newMaterial(0.f);
            newID = store.addMaterial(newMaterial);
            break;
        }
        case MaterialType::Mirror: {
            Mirror newMaterial;
            newID = store.addMaterial(newMaterial);
            break;
        }
        case MaterialType::Glass: {
            Glass newMaterial(1.4f);
            newID = store.addMaterial(newMaterial);
            break;
        }
        }

        MaterialIndex newIndex = store.indexAt(newID);
        store.updateIndex(materialID, newIndex);
    }

    Vec3 getColor(int materialID) const {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            return getLambertians()[materialIndex.index].getAlbedo();
        } else {
            return Vec3(0.f);
        }
    }
    Vec3 getEmit(int materialID) const {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            return getLambertians()[materialIndex.index].getEmit();
        } else {
            return Vec3(0.f);
        }
    }

    void setColor(int materialID, Vec3 color) {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            Lambertian material = getLambertians()[materialIndex.index];
            material.setAlbedo(color);
            m_sceneData.materialStore.updateMaterial(materialIndex.index, material);
        }
    }
    void setEmit(int materialID, Vec3 color) {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            Lambertian material = getLambertians()[materialIndex.index];
            material.setEmit(color);
            m_sceneData.materialStore.updateMaterial(materialIndex.index, material);
        }
    }

    float getIOR(int materialID) const {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Glass) {
            return getGlasses()[materialIndex.index].getIOR();
        }
        return -1.f;
    }
    void setIOR(int materialID, float ior) {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Glass) {
            Glass material = getGlasses()[materialIndex.index];
            material.setIOR(ior);
            m_sceneData.materialStore.updateMaterial(materialIndex.index, material);
        }
    }

    MaterialType getMaterialType(int materialID) const {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);
        return materialIndex.materialType;
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
