#pragma once

#include <iostream>
#include <vector>

#include "camera.h"
#include "materials/mirror.h"
#include "materials/material.h"
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
        SceneData sceneData
    ) : m_camera(camera),
        m_sceneData(sceneData),
        m_maxDepth(defaultMaxDepth),
        m_nextEventEstimation(true)
    {}

    void init();
    void update();

    const Camera &getCamera() const { return m_camera; }
    void setCamera(const Camera &camera) { m_camera = camera; }

    const SceneData &getSceneData() const { return m_sceneData; }

    const Material *getLambertiansData() const { return m_lambertians.data(); }
    size_t getLambertiansSize() const { return m_lambertians.size() * sizeof(Material); }

    const Mirror *getMirrorsData() const { return m_mirrors.data(); }
    size_t getMirrorsSize() const { return m_mirrors.size() * sizeof(Mirror); }

    Vec3 getColor(int materialID) const {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            return m_lambertians[materialIndex.index].getAlbedo();
        } else {
            return Vec3(0.f);
        }
    }
    Vec3 getEmit(int materialID) const {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            return m_lambertians[materialIndex.index].getEmit();
        } else {
            return Vec3(0.f);
        }
    }

    void setColor(int materialID, Vec3 color) {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            m_lambertians[materialIndex.index].setAlbedo(color);
        }
    }
    void setEmit(int materialID, Vec3 color) {
        MaterialIndex materialIndex = m_sceneData.materialStore.indexAt(materialID);

        if (materialIndex.materialType == MaterialType::Lambertian) {
            m_lambertians[materialIndex.index].setEmit(color);
        }
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
    std::vector<Material> m_materials;
    std::vector<Material> m_lambertians;
    std::vector<Mirror> m_mirrors;
    int m_maxDepth;
    bool m_nextEventEstimation;
};

}


namespace rays { namespace SceneParameters {

SceneData getSceneData(int index);
Camera getCamera(int index, Resolution resolution);

} }
