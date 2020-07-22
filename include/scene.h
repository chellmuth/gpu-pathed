#pragma once

#include <vector>

#include "material.h"
#include "primitive.h"
#include "vec3.h"

namespace rays {

constexpr float defaultLightPosition = -0.6f;

constexpr int primitiveCount = 4;
constexpr int materialCount = 3;

__global__ void createWorld(
    Primitive **primitives,
    Material *materials,
    PrimitiveList *world,
    float lightPosition,
    bool update
);

class Scene {
public:
    Scene(Vec3 color) : m_color(color) {}

    void init();
    void update();

    const Material *getMaterialsData() const { return m_materials.data(); }
    const Material &getMaterial(int materialIndex) const { return m_materials[materialIndex]; }
    size_t getMaterialsSize() const { return m_materials.size() * sizeof(Material); }

    void setColor(int materialIndex, Vec3 color) {
        m_materials[materialIndex] = color;
        /* m_color = color; */
        /* update(); */
    }

private:
    Vec3 m_color;
    std::vector<Material> m_materials;
};

}
