#include "cuda_globals.h"

#include "macro_helper.h"
#include "scene.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

void CUDAGlobals::mallocCamera()
{
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera)));
}

void CUDAGlobals::copyCamera(const Camera &camera)
{
    checkCudaErrors(cudaMemcpy(
        d_camera,
        &camera,
        sizeof(Camera),
        cudaMemcpyHostToDevice
    ));
}

__global__ static void initWorldKernel(
    World *world,
    Triangle *triangles,
    int triangleSize,
    Sphere *spheres,
    int sphereSize,
    int *lightIndices,
    int lightIndexSize,
    EnvironmentLight *environmentLight,
    MaterialLookup *materialLookup
) {
    if (blockIdx.x != 0 || blockIdx.y != 0) { return; }
    if (threadIdx.x != 0 || threadIdx.y != 0) { return; }

    *world = World(
        triangles,
        triangleSize,
        spheres,
        sphereSize,
        lightIndices,
        lightIndexSize,
        environmentLight,
        materialLookup
    );
}

void CUDAGlobals::initMaterials(const SceneData &sceneData)
{
    checkCudaErrors(cudaMalloc((void **)&d_materialLookup, sizeof(MaterialLookup)));

    m_materialLookup.mallocMaterials(sceneData);
    m_materialLookup.copyMaterials(sceneData);

    checkCudaErrors(cudaMemcpy(
        d_materialLookup,
        &m_materialLookup,
        sizeof(MaterialLookup),
        cudaMemcpyHostToDevice
    ));
}

void CUDAGlobals::updateMaterials(const SceneData &sceneData)
{
    m_materialLookup.freeMaterials();
    m_materialLookup.mallocMaterials(sceneData);
    m_materialLookup.copyMaterials(sceneData);

    checkCudaErrors(cudaMemcpy(
        d_materialLookup,
        &m_materialLookup,
        sizeof(MaterialLookup),
        cudaMemcpyHostToDevice
    ));
}

void CUDAGlobals::mallocWorld(const SceneData &sceneData)
{
    const int triangleSize = sceneData.triangles.size();
    const int sphereSize = sceneData.spheres.size();
    const int lightIndexSize = sceneData.lightIndices.size();

    checkCudaErrors(cudaMalloc((void **)&d_triangles, triangleSize * sizeof(Triangle)));
    checkCudaErrors(cudaMalloc((void **)&d_spheres, sphereSize * sizeof(Sphere)));
    checkCudaErrors(cudaMalloc((void **)&d_lightIndices, lightIndexSize * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_environmentLight, sizeof(EnvironmentLight)));

    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(World)));

    initWorldKernel<<<1, 1>>>(
        d_world,
        d_triangles,
        triangleSize,
        d_spheres,
        sphereSize,
        d_lightIndices,
        lightIndexSize,
        d_environmentLight,
        d_materialLookup
    );
    checkCudaErrors(cudaDeviceSynchronize());

    const auto &environmentLightParams = sceneData.environmentLightParams;
    m_environmentLight = environmentLightParams.createEnvironmentLight();
}

void CUDAGlobals::copySceneData(const SceneData &sceneData)
{
    checkCudaErrors(cudaMemcpy(
        d_triangles,
        sceneData.triangles.data(),
        sceneData.triangles.size() * sizeof(Triangle),
        cudaMemcpyHostToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_spheres,
        sceneData.spheres.data(),
        sceneData.spheres.size() * sizeof(Sphere),
        cudaMemcpyHostToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_lightIndices,
        sceneData.lightIndices.data(),
        sceneData.lightIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_environmentLight,
        &m_environmentLight,
        sizeof(EnvironmentLight),
        cudaMemcpyHostToDevice
    ));
}

}
