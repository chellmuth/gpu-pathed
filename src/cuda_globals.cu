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

__global__ static void updateMaterialLookup(
    MaterialLookup *materialLookup,
    MaterialIndex *materialIndices,
    Lambertian *lambertians,
    Mirror *mirrors,
    Glass *glasses
) {
    if (blockIdx.x != 0 || blockIdx.y != 0) { return; }
    if (threadIdx.x != 0 || threadIdx.y != 0) { return; }

    materialLookup->indices = materialIndices;
    materialLookup->lambertians = lambertians;
    materialLookup->mirrors = mirrors;
    materialLookup->glasses = glasses;
}

__global__ static void initWorldKernel(
    PrimitiveList *world,
    Triangle *triangles,
    int triangleSize,
    Sphere *spheres,
    int sphereSize,
    int *lightIndices,
    int lightIndexSize,
    MaterialLookup *materialLookup
) {
    if (blockIdx.x != 0 || blockIdx.y != 0) { return; }
    if (threadIdx.x != 0 || threadIdx.y != 0) { return; }

    *world = PrimitiveList(
        triangles,
        triangleSize,
        spheres,
        sphereSize,
        lightIndices,
        lightIndexSize,
        materialLookup
    );
}

void CUDAGlobals::mallocMaterials(const SceneData &sceneData)
{
    const int lambertianSize = sceneData.materialStore.getLambertians().size();
    const int mirrorSize = sceneData.materialStore.getMirrors().size();
    const int glassSize = sceneData.materialStore.getGlasses().size();

    const std::vector<MaterialIndex> &indices = sceneData.materialStore.getIndices();

    checkCudaErrors(cudaMalloc((void **)&d_materialIndices, indices.size() * sizeof(MaterialIndex)));
    checkCudaErrors(cudaMalloc((void **)&d_lambertians, lambertianSize * sizeof(Lambertian)));
    checkCudaErrors(cudaMalloc((void **)&d_mirrors, mirrorSize * sizeof(Mirror)));
    checkCudaErrors(cudaMalloc((void **)&d_glasses, glassSize * sizeof(Glass)));
}

void CUDAGlobals::copyMaterials(const SceneData &sceneData)
{
    const std::vector<MaterialIndex> indices = sceneData.materialStore.getIndices();
    checkCudaErrors(cudaMemcpy(
        d_materialIndices,
        indices.data(),
        indices.size() * sizeof(MaterialIndex),
        cudaMemcpyHostToDevice
    ));

    const std::vector<Lambertian> &lambertians = sceneData.materialStore.getLambertians();
    checkCudaErrors(cudaMemcpy(
        d_lambertians,
        lambertians.data(),
        lambertians.size() * sizeof(Lambertian),
        cudaMemcpyHostToDevice
    ));

    const std::vector<Mirror> &mirrors = sceneData.materialStore.getMirrors();
    checkCudaErrors(cudaMemcpy(
        d_mirrors,
        mirrors.data(),
        mirrors.size() * sizeof(Mirror),
        cudaMemcpyHostToDevice
    ));

    const std::vector<Glass> &glasses = sceneData.materialStore.getGlasses();
    checkCudaErrors(cudaMemcpy(
        d_glasses,
        glasses.data(),
        glasses.size() * sizeof(Glass),
        cudaMemcpyHostToDevice
    ));

    updateMaterialLookup<<<1, 1>>>(
        d_materialLookup,
        d_materialIndices,
        d_lambertians,
        d_mirrors,
        d_glasses
    );

    checkCudaErrors(cudaDeviceSynchronize());
}

void CUDAGlobals::freeMaterials()
{
    checkCudaErrors(cudaFree(d_materialIndices));
    checkCudaErrors(cudaFree(d_lambertians));
    checkCudaErrors(cudaFree(d_mirrors));
    checkCudaErrors(cudaFree(d_glasses));
}

void CUDAGlobals::mallocWorld(const SceneData &sceneData)
{
    const int triangleSize = sceneData.triangles.size();
    const int sphereSize = sceneData.spheres.size();
    const int lightIndexSize = sceneData.lightIndices.size();

    checkCudaErrors(cudaMalloc((void **)&d_triangles, triangleSize * sizeof(Triangle)));
    checkCudaErrors(cudaMalloc((void **)&d_spheres, sphereSize * sizeof(Sphere)));
    checkCudaErrors(cudaMalloc((void **)&d_lightIndices, lightIndexSize * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_materialLookup, sizeof(MaterialLookup)));

    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(PrimitiveList)));

    initWorldKernel<<<1, 1>>>(
        d_world,
        d_triangles,
        triangleSize,
        d_spheres,
        sphereSize,
        d_lightIndices,
        lightIndexSize,
        d_materialLookup
    );
    checkCudaErrors(cudaDeviceSynchronize());
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
}

}
