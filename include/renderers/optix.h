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
    Optix() : m_currentSamples(0) {}

    void init(int width, int height, const Scene &scene);
    uchar4 *launch();
    int getCurrentSamples() const { return m_currentSamples; }

private:
    int m_width;
    int m_height;
    Params m_params;

    int m_currentSamples;

    uchar4 *d_image;
    Vec3 *d_passRadiances;
    Vec3 *d_radiances;
    CUdeviceptr d_param;

    OptixTraversableHandle m_gasHandle;
    OptixPipeline m_pipeline;
    OptixShaderBindingTable m_sbt;
};

}
