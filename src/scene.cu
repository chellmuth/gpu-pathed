#include "scene.h"

#include <iostream>

#include "material.h"
#include "sphere.h"
#include "triangle.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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

static Vec3 rotateY(Vec3 vector, float theta)
{
    return Vec3(
        cos(theta) * vector.x() - sin(theta) * vector.z(),
        vector.y(),
        sin(theta) * vector.x() + cos(theta) * vector.z()
    );
}

void copyGeometry(
    Triangle *d_triangles,
    Sphere *d_spheres,
    Material *d_materials,
    PrimitiveList *d_world,
    float lightPosition
) {
    Sphere localSpheres[sphereCount] = {
        Sphere(
            Vec3(0.f, 0.f, -1.f),
            0.8f,
            0
        ),
        Sphere(
            Vec3(0.f, -100.8f, -1.f),
            100.f,
            2
        )
    };

    const float theta = lightPosition * M_PI;
    Triangle localTriangles[triangleCount] = {
        Triangle(
            rotateY(Vec3(-0.5f, 0.6f, -2.f), theta),
            rotateY(Vec3(0.3f, 0.6f, -2.f), theta),
            rotateY(Vec3(-0.5f, 1.2f, -1.8f), theta),
            1
        ),
        Triangle(
            rotateY(Vec3(0.3f, 0.6f, -2.f), theta),
            rotateY(Vec3(0.3f, 1.2f, -1.8f), theta),
            rotateY(Vec3(-0.5f, 1.2f, -1.8f), theta),
            1
        )
    };

    checkCudaErrors(cudaMemcpy(
        d_triangles,
        &localTriangles,
        triangleCount * sizeof(Triangle),
        cudaMemcpyHostToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_spheres,
        &localSpheres,
        sphereCount * sizeof(Sphere),
        cudaMemcpyHostToDevice
    ));
}

}

#undef checkCudaErrors
