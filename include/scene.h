#pragma once

#include <vector>

#include "camera.h"
#include "material.h"
#include "primitive.h"
#include "scene_data.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"

namespace rays {

constexpr float defaultLightPosition = -0.6f;

namespace SceneParameters {

inline SceneData getSceneData(int index) {
    if (index == 0) {
        std::string sceneFilename("../scenes/cornell-box/CornellBox-Original.obj");
        ObjParser objParser(sceneFilename);
        return SceneAdapter::createSceneData(objParser);
    } else {
        return SceneAdapter::createSceneData(defaultLightPosition);
    }
}

inline Camera getCamera(int index, Resolution resolution) {
    if (index == 0) {
        return Camera(
            Vec3(0.f, 1.f, 6.8f),
            Vec3(0.f, 1.f, 0.f),
            19.5f / 180.f * M_PI,
            resolution
        );
    } else {
        return Camera(
            Vec3(0.f, 0.3f, 5.f),
            Vec3(0.f),
            30.f / 180.f * M_PI,
            resolution
        );
    }
}

}

class Scene {
public:
    Scene(
        Camera &camera,
        SceneData sceneData
    ) : m_camera(camera),
        m_sceneData(sceneData)
    {}

    void init();
    void update();

    Camera &getCamera() { return m_camera; }
    SceneData &getSceneData() { return m_sceneData; }

    const Material *getMaterialsData() const { return m_materials.data(); }
    const Material &getMaterial(int materialIndex) const { return m_materials[materialIndex]; }
    size_t getMaterialsSize() const { return m_materials.size() * sizeof(Material); }

    void setColor(int materialIndex, Vec3 color) {
        m_materials[materialIndex].setAlbedo(color);
    }
    void setEmit(int materialIndex, Vec3 color) {
        m_materials[materialIndex].setEmit(color);
    }

private:
    Camera m_camera;
    SceneData m_sceneData;
    std::vector<Material> m_materials;
};

}
