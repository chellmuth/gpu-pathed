#include "materials/material_lookup.h"

#include <vector>

#include <cuda_runtime.h>

#include "macro_helper.h"

#define checkCUDA(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

void MaterialLookup::mallocMaterials(const SceneData &sceneData)
{
    int lambertianSize = 0;
    int mirrorSize = 0;
    int glassSize = 0;

    for (const auto &param : sceneData.materialParams) {
        switch (param->getMaterialType()) {
        case MaterialType::Lambertian: {
            lambertianSize += 1;
            break;
        }
        case MaterialType::Mirror: {
            mirrorSize += 1;
            break;
        }
        case MaterialType::Glass: {
            glassSize += 1;
            break;
        }
        }
    }

    checkCUDA(cudaMalloc((void **)&indices, sceneData.materialParams.size() * sizeof(MaterialIndex)));
    checkCUDA(cudaMalloc((void **)&lambertians, lambertianSize * sizeof(Lambertian)));
    checkCUDA(cudaMalloc((void **)&mirrors, mirrorSize * sizeof(Mirror)));
    checkCUDA(cudaMalloc((void **)&glasses, glassSize * sizeof(Glass)));
}

void MaterialLookup::copyMaterials(const SceneData &sceneData)
{
    std::vector<MaterialIndex> indicesLocal;
    std::vector<Lambertian> lambertiansLocal;
    std::vector<Mirror> mirrorsLocal;
    std::vector<Glass> glassesLocal;

    for (const auto &param : sceneData.materialParams) {
        switch (param->getMaterialType()) {
        case MaterialType::Lambertian: {
            Lambertian lambertian(*param);
            lambertiansLocal.push_back(lambertian);

            indicesLocal.push_back({MaterialType::Lambertian, lambertiansLocal.size() - 1});
            break;
        }
        case MaterialType::Mirror: {
            Mirror mirror(*param);
            mirrorsLocal.push_back(mirror);

            indicesLocal.push_back({MaterialType::Mirror, mirrorsLocal.size() - 1});
            break;
        }
        case MaterialType::Glass: {
            Glass glass(*param);
            glassesLocal.push_back(glass);

            indicesLocal.push_back({MaterialType::Glass, glassesLocal.size() - 1});
            break;
        }
        }
    }

    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(indices),
        indicesLocal.data(),
        indicesLocal.size() * sizeof(MaterialIndex),
        cudaMemcpyHostToDevice
    ));

    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(lambertians),
        lambertiansLocal.data(),
        lambertiansLocal.size() * sizeof(Lambertian),
        cudaMemcpyHostToDevice
    ));

    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(mirrors),
        mirrorsLocal.data(),
        mirrorsLocal.size() * sizeof(Mirror),
        cudaMemcpyHostToDevice
    ));

    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(glasses),
        glassesLocal.data(),
        glassesLocal.size() * sizeof(Glass),
        cudaMemcpyHostToDevice
    ));
}

void MaterialLookup::freeMaterials()
{
    checkCUDA(cudaFree(indices));
    checkCUDA(cudaFree(lambertians));
    checkCUDA(cudaFree(mirrors));
    checkCUDA(cudaFree(glasses));
}

}
