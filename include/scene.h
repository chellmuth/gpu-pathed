#pragma once

#include <vector>

#include "material.h"
#include "primitive.h"
#include "scene_data.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"

namespace rays {

constexpr float defaultLightPosition = -0.6f;

class Scene {
public:
    Scene(SceneData sceneData)
        : m_sceneData(sceneData)
    {}

    void init();
    void update();

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
    SceneData m_sceneData;
    std::vector<Material> m_materials;
};

}
