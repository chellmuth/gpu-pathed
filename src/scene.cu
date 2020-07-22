#include "scene.h"

#include "material.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

void Scene::init()
{
    update();
}

void Scene::update()
{
    m_materials.clear();

    m_materials.push_back(Material(Vec3(0.45098f, 0.823529f, 0.0862745f)));
    m_materials.push_back(Material(Vec3(1.f, 0.2f, 1.f), Vec3(14.f, 14.f, 14.f)));
    m_materials.push_back(Material(Vec3(1.f, 1.f, 1.f)));
}

__device__ static Vec3 rotateY(Vec3 vector, float theta)
{
    return Vec3(
        cos(theta) * vector.x() - sin(theta) * vector.z(),
        vector.y(),
        sin(theta) * vector.x() + cos(theta) * vector.z()
    );
}

__global__ void createWorld(
    Triangle **triangles,
    Sphere **spheres,
    Material *materials,
    PrimitiveList *world,
    float lightPosition,
    bool update
) {
    if (update) {
        for (int i = 0; i < triangleCount; i++) {
            delete(triangles[i]);
        }
        for (int i = 0; i < sphereCount; i++) {
            delete(spheres[i]);
        }
        delete(world);
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        size_t sphereSize = 0;
        size_t triangleSize = 0;

        spheres[sphereSize++] = new Sphere(
            Vec3(0.f, 0.f, -1.f),
            0.8f,
            0
        );

        const float theta = lightPosition * M_PI;
        triangles[triangleSize++] = new Triangle(
            rotateY(Vec3(-0.5f, 0.6f, -2.f), theta),
            rotateY(Vec3(0.3f, 0.6f, -2.f), theta),
            rotateY(Vec3(-0.5f, 1.2f, -1.8f), theta),
            1
        );
        triangles[triangleSize++] = new Triangle(
            rotateY(Vec3(0.3f, 0.6f, -2.f), theta),
            rotateY(Vec3(0.3f, 1.2f, -1.8f), theta),
            rotateY(Vec3(-0.5f, 1.2f, -1.8f), theta),
            1
        );

        spheres[sphereSize++] = new Sphere(
            Vec3(0.f, -100.8f, -1.f),
            100.f,
            2
        );

        *world = PrimitiveList(
            triangles,
            triangleSize,
            spheres,
            sphereSize,
            materials,
            materialCount
        );
    }
}

}
