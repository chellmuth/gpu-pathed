#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include "camera.h"
#include "materials/material.h"
#include "scene.h"
#include "vec3.h"

namespace rays {

struct Params
{
    Vec3 *passRadiances;
    int launchCount;
    int samplesPerPass;
    unsigned int width;
    unsigned int height;
    Camera camera;
    int maxDepth;
    bool useNextEventEstimation;
    MaterialLookup *materialLookup;
    Triangle *triangles;
    int *lightIndices;
    int lightIndexSize;
    OptixTraversableHandle handle;
};

struct RayGenData {};
struct MissData {};
struct HitGroupData {
    int materialID;
};

class Optix {
public:
    void init(int width, int height, const Scene &scene);
    uchar4 *launch(int spp, int currentSamples);

    void updateCamera(const Scene &scene);
    void updateMaxDepth(const Scene &scene);
    void updateNextEventEstimation(const Scene &scene);
    void updateMaterials(const Scene &scene);

private:
    void mallocMaterials(const Scene &scene);
    void copyMaterials(const Scene &scene);
    void freeMaterials();

    MaterialLookup m_materialLookup;

    int m_width;
    int m_height;
    Params m_params;

    uchar4 *d_image;
    Vec3 *d_passRadiances;
    Vec3 *d_radiances;
    Material *d_materials;
    MaterialLookup *d_materialLookup;
    CUdeviceptr d_param;

    OptixTraversableHandle m_gasHandle;
    OptixPipeline m_pipeline;
    OptixShaderBindingTable m_sbt;
};

}
