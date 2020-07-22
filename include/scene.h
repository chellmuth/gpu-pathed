#pragma once

#include <vector>

#include "material.h"
#include "primitive.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"

namespace rays {

constexpr float defaultLightPosition = -0.6f;

constexpr int triangleCount = 2;
constexpr int sphereCount = 2;
constexpr int materialCount = 3;

__global__ void createWorld(
    Triangle **triangles,
    Sphere **spheres,
    Material *materials,
    PrimitiveList *world,
    float lightPosition,
    bool update
);

class Scene {
public:
    void init();
    void update();

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
    std::vector<Material> m_materials;
};

}
