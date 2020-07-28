#pragma once

#include <iostream>
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
constexpr int defaultMaxDepth = 3;

class Scene {
public:
    Scene(
        Camera &camera,
        SceneData sceneData
    ) : m_camera(camera),
        m_sceneData(sceneData),
        m_maxDepth(defaultMaxDepth)
    {}

    void init();
    void update();

    const Camera &getCamera() const { return m_camera; }
    void setCamera(const Camera &camera) { m_camera = camera; }

    const SceneData &getSceneData() const { return m_sceneData; }

    const Material *getMaterialsData() const { return m_materials.data(); }
    const Material &getMaterial(int materialIndex) const { return m_materials[materialIndex]; }
    size_t getMaterialsSize() const { return m_materials.size() * sizeof(Material); }

    void setColor(int materialIndex, Vec3 color) {
        m_materials[materialIndex].setAlbedo(color);
    }
    void setEmit(int materialIndex, Vec3 color) {
        m_materials[materialIndex].setEmit(color);
    }

    int getMaxDepth() const {
        return m_maxDepth;
    }

    void setMaxDepth(int maxDepth) {
        m_maxDepth = maxDepth;
    }

private:
    Camera m_camera;
    SceneData m_sceneData;
    std::vector<Material> m_materials;
    int m_maxDepth;
};

}


namespace rays { namespace SceneParameters {

SceneData getSceneData(int index);
Camera getCamera(int index, Resolution resolution);

} }
