#include "scene.h"

#include <iostream>

#include "material.h"
#include "scene_data.h"
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
    SceneData sceneData = SceneAdapter::createSceneData(defaultLightPosition);

    m_materials = sceneData.materials;
}

void copyGeometry(
    Triangle *d_triangles,
    Sphere *d_spheres,
    Material *d_materials,
    PrimitiveList *d_world,
    float lightPosition
) {
    SceneData sceneData = SceneAdapter::createSceneData(lightPosition);

    checkCudaErrors(cudaMemcpy(
        d_triangles,
        sceneData.triangles.data(),
        triangleCount * sizeof(Triangle),
        cudaMemcpyHostToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_spheres,
        sceneData.spheres.data(),
        sphereCount * sizeof(Sphere),
        cudaMemcpyHostToDevice
    ));
}

}

#undef checkCudaErrors
