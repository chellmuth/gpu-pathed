#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include "camera.h"
#include "material.h"
#include "scene.h"
#include "vec3.h"

namespace rays {

struct Params
{
    Vec3 *passRadiances;
    int launchCount;
    unsigned int width;
    unsigned int height;
    Camera camera;
    int maxDepth;
    Material *materials;
    Triangle *primitives;
    OptixTraversableHandle handle;
};

struct RayGenData {};
struct MissData {};
struct HitGroupData {
    int materialIndex;
};

class Optix {
public:
    void init(int width, int height, const Scene &scene);
    uchar4 *launch(int currentSamples);

    void updateCamera(const Scene &scene);
    void updateMaterials(const Scene &scene);
    void updateMaxDepth(const Scene &scene);

private:
    int m_width;
    int m_height;
    Params m_params;

    uchar4 *d_image;
    Vec3 *d_passRadiances;
    Vec3 *d_radiances;
    Material *d_materials;
    CUdeviceptr d_param;

    OptixTraversableHandle m_gasHandle;
    OptixPipeline m_pipeline;
    OptixShaderBindingTable m_sbt;
};

}