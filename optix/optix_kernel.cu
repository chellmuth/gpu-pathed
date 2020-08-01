#include <optix.h>

#include "frame.h"
#include "material.h"
#include "ray.h"
#include "renderers/float3_helpers.h"
#include "renderers/optix.h"
#include "renderers/payload_helpers.h"
#include "renderers/random.h"
#include "sampler.h"
#include "triangle.h"
#include "vec3.h"
#include "world_frame.h"

extern "C" {
    __constant__ rays::Params params;
}

struct Intersection {
    rays::Vec3 point;
    rays::Vec3 normal;
    rays::Vec3 woLocal;
    rays::Frame frame;

    __device__ bool isFront() const {
        return woLocal.z() >= 0.f;
    }
};

struct PerRayData {
    bool done;
    float3 beta;
    const rays::Material *material;
    float3 point;
    Intersection intersection;
    int pad;
};

__forceinline__ __device__ static PerRayData *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData *>(rays::unpackPointer(u0, u1));
}

__forceinline__ __device__ static rays::Vec3 direct(
    const Intersection &intersection,
    const rays::Material &hitMaterial,
    unsigned int &seed
) {
    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);
    const float xi3 = rnd(seed);

    const rays::LightSample lightSample = rays::Sampler::sampleDirectLights(
        intersection.point,
        make_float3(xi1, xi2, xi3),
        params.lightIndices,
        params.lightIndexSize,
        params.triangles
    );

    const rays::Vec3 lightDirection = (lightSample.point - intersection.point);
    const rays::Vec3 wiWorld = normalized(lightDirection);
    const rays::Ray shadowRay(intersection.point, wiWorld);

    PerRayData prd;
    prd.done = false;

    unsigned int p0, p1;
    rays::packPointer(&prd, p0, p1);
    optixTrace(
        params.handle,
        vec3_to_float3(shadowRay.origin()),
        vec3_to_float3(shadowRay.direction()),
        1e-4,
        lightDirection.length() - 2e-4,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        1,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1
    );

    const bool isHit = !prd.done;
    if (!isHit) {
        const rays::Vec3 wi = intersection.frame.toLocal(wiWorld);
        const float pdf = lightSample.solidAnglePDF(intersection.point);
        const rays::Material &emitMaterial = params.materials[lightSample.materialIndex];
        const rays::Vec3 lightContribution = rays::Vec3(1.f)
            * emitMaterial.getEmit()
            * hitMaterial.f(intersection.woLocal, wi)
            * rays::WorldFrame::absCosTheta(intersection.normal, wiWorld)
            / pdf;

        return lightContribution;
    } else {
        return rays::Vec3(0.f);
    }
}

__forceinline__ __device__ static rays::Vec3 LiNEE(
    const rays::Ray &cameraRay,
    unsigned int &seed
) {
    rays::Vec3 result(0.f);

    if (params.maxDepth == 0) { return result; }

    const rays::Vec3 origin = cameraRay.origin();
    const rays::Vec3 direction = cameraRay.direction();

    rays::Vec3 beta(1.f);

    PerRayData prd;
    prd.done = false;

    unsigned int p0, p1;
    rays::packPointer(&prd, p0, p1);
    optixTrace(
        params.handle,
        make_float3(origin.x(), origin.y(), origin.z()),
        make_float3(direction.x(), direction.y(), direction.z()),
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        1,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1
    );

    if (!prd.done) {
        const Intersection &intersection = prd.intersection;
        const rays::Material &material = *prd.material;

        if (intersection.isFront()) {
            result += material.getEmit();
        }
    }

    for (int path = 1; path < params.maxDepth; path++) {
        if (prd.done) { break; }

        const Intersection &intersection = prd.intersection;
        const rays::Material &material = *prd.material;

        result += direct(intersection, material, seed) * beta;

        const float xi1 = rnd(seed);
        const float xi2 = rnd(seed);
        float pdf;
        const rays::Vec3 sample = material.sample(&pdf, xi1, xi2);

        const rays::Frame &frame = intersection.frame;
        const rays::Vec3 bounceWorld = normalized(frame.toWorld(sample));

        const float3 normal = vec3_to_float3(intersection.normal);
        beta *= material.f(intersection.woLocal, sample) * sample.z() / pdf;

        rays::packPointer(&prd, p0, p1);
        optixTrace(
            params.handle,
            vec3_to_float3(intersection.point),
            vec3_to_float3(bounceWorld),
            1e-4,
            1e16f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            p0, p1
        );
    }

    return result;
}

__forceinline__ __device__ static rays::Vec3 LiNaive(
    const rays::Ray &cameraRay,
    unsigned int &seed
) {
    rays::Vec3 result(0.f);

    if (params.maxDepth == 0) { return result; }

    const rays::Vec3 origin = cameraRay.origin();
    const rays::Vec3 direction = cameraRay.direction();

    rays::Vec3 beta(1.f);

    PerRayData prd;
    prd.done = false;

    unsigned int p0, p1;
    rays::packPointer(&prd, p0, p1);
    optixTrace(
        params.handle,
        make_float3(origin.x(), origin.y(), origin.z()),
        make_float3(direction.x(), direction.y(), direction.z()),
        0.0f,                // Min intersection distance
        1e16f,               // Max intersection distance
        0.0f,                // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        1,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1
    );

    if (!prd.done) {
        const Intersection &intersection = prd.intersection;
        const rays::Material &material = *prd.material;

        if (intersection.isFront()) {
            result += material.getEmit();
        }
    }

    for (int path = 1; path < params.maxDepth; path++) {
        if (prd.done) { break; }

        const Intersection &intersection = prd.intersection;
        const rays::Material &material = *prd.material;

        const float xi1 = rnd(seed);
        const float xi2 = rnd(seed);
        float pdf;
        const rays::Vec3 sample = material.sample(&pdf, xi1, xi2);

        const rays::Frame &frame = intersection.frame;
        const rays::Vec3 bounceWorld = normalized(frame.toWorld(sample));

        const float3 normal = vec3_to_float3(intersection.normal);
        beta *= material.f(intersection.woLocal, sample) * sample.z() / pdf;

        rays::packPointer(&prd, p0, p1);
        optixTrace(
            params.handle,
            vec3_to_float3(intersection.point),
            vec3_to_float3(bounceWorld),
            1e-4,
            1e16f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            p0, p1
        );

        if (prd.done) { break; }

        const rays::Material &bounceMaterial = *prd.material;
        if (intersection.isFront()) {
            result += bounceMaterial.getEmit() * beta;
        }
    }

    return result;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    unsigned int seed = tea<4>(idx.y * params.width + idx.x, params.launchCount);

    const int row = idx.y;
    const int col = idx.x;

    rays::Vec3 result(0.f);

    for (int i = 0; i < params.samplesPerPass; i++) {
        const rays::Ray cameraRay = params.camera.generateRay(
            row, col,
            sample_float2(seed)
        );
        if (params.useNextEventEstimation) {
            result += LiNEE(cameraRay, seed);
        } else {
            result += LiNaive(cameraRay, seed);
        }
    }

    params.passRadiances[idx.y * params.width + idx.x] = result / params.samplesPerPass;
}

extern "C" __global__ void __miss__ms()
{
    PerRayData *prd = getPRD();
    prd->done = true;
}

extern "C" __global__ void __closesthit__ch()
{
    PerRayData *prd = getPRD();
    prd->done = false;

    const int primitiveIndex = optixGetPrimitiveIndex();
    const rays::Triangle &triangle = params.triangles[primitiveIndex];

    rays::HitGroupData* hitgroupData = reinterpret_cast<rays::HitGroupData *>(optixGetSbtDataPointer());
    const int materialIndex = hitgroupData->materialIndex;

    const rays::Material &material = params.materials[materialIndex];

    Intersection intersection;
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    intersection.point = triangle.interpolate(u, v);
    intersection.normal = triangle.interpolateNormal(u, v);
    intersection.frame = rays::Frame(intersection.normal);
    intersection.woLocal = intersection.frame.toLocal(
        float3_to_vec3(-optixGetWorldRayDirection())
    );

    prd->intersection = intersection;
    prd->material = &material;
}
