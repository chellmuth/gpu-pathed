#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include "core/camera.h"
#include "core/vec3.h"
#include "lights/environment_light.h"
#include "materials/lambertian.h"
#include "scene.h"

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
    EnvironmentLight environmentLight;
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
    uchar4 *launch(Vec3 *d_radiances, int spp, int currentSamples);

    void updateCamera(const Scene &scene);
    void updateMaxDepth(const Scene &scene);
    void updateNextEventEstimation(const Scene &scene);
    void updateMaterials(const Scene &scene);

private:
    void initMaterials(const SceneData &sceneData);

    MaterialLookup m_materialLookup;

    int m_width;
    int m_height;
    Params m_params;

    uchar4 *d_image;
    Vec3 *d_passRadiances;
    Lambertian *d_materials;
    MaterialLookup *d_materialLookup;
    CUdeviceptr d_param;

    OptixTraversableHandle m_gasHandle;
    OptixPipeline m_pipeline;
    OptixShaderBindingTable m_sbt;
};

}
