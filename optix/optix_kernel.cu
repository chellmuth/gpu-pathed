#include <optix.h>

#include "frame.h"
#include "intersection.h"
#include "materials/bsdf.h"
#include "materials/bsdf_sample.h"
#include "materials/lambertian.h"
#include "materials/types.h"
#include "core/ray.h"
#include "renderers/float3_helpers.h"
#include "renderers/optix.h"
#include "renderers/payload_helpers.h"
#include "renderers/random.h"
#include "lights/sampler.h"
#include "tangent_frame.h"
#include "primitives/triangle.h"
#include "core/vec3.h"
#include "world_frame.h"

extern "C" {
    __constant__ rays::Params params;
}

struct PerRayData {
    bool done;
    float3 beta;
    int materialID;
    float3 point;
    rays::Intersection intersection;
    int pad;
};

__forceinline__ __device__ static rays::Vec3 f(
    int materialID,
    const rays::Vec3 &wo,
    const rays::Vec3 &wi
) {
    rays::BSDF bsdf(params.materialLookup, materialID);
    return bsdf.f(wo, wi);
}

__forceinline__ __device__ static rays::BSDFSample sample(
    int materialID,
    const rays::Intersection &intersection,
    unsigned int &seed
) {
    rays::BSDF bsdf(params.materialLookup, materialID);
    return bsdf.sample(intersection, seed);
}

__forceinline__ __device__ static rays::Vec3 getEmit(int materialID)
{
    rays::BSDF bsdf(params.materialLookup, materialID);
    return bsdf.getEmit();
}

__forceinline__ __device__ static PerRayData *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData *>(rays::unpackPointer(u0, u1));
}

__forceinline__ __device__ static rays::Vec3 directSampleBSDF(
    const rays::Intersection &intersection,
    const rays::BSDFSample &bsdfSample,
    unsigned int &seed
) {
    if (!bsdfSample.isDelta) { return rays::Vec3(0.f); }

    const float bsdfWeight = 1.f;

    const rays::Vec3 bounceDirection = intersection.frame.toWorld(bsdfSample.wiLocal);
    const rays::Ray bounceRay(intersection.point, bounceDirection);

    PerRayData prd;
    prd.done = false;

    unsigned int p0, p1;
    rays::packPointer(&prd, p0, p1);
    optixTrace(
        params.handle,
        vec3_to_float3(bounceRay.origin()),
        vec3_to_float3(bounceRay.direction()),
        1e-4,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // sbt offset, stride, miss index
        p0, p1
    );

    rays::Vec3 emit(0.f);
    const bool hit = !prd.done;
    if (hit) {
        emit = getEmit(prd.materialID);
    } else {
        emit = params.environmentLight.getEmit(bounceRay.direction());
    }

    if (emit.isZero()) { return rays::Vec3(0.f); }

    return emit
        * bsdfWeight
        * bsdfSample.f
        * rays::TangentFrame::absCosTheta(bsdfSample.wiLocal)
        / bsdfSample.pdf;
}

__forceinline__ __device__ static rays::Vec3 directSampleLights(
    const rays::Intersection &intersection,
    int materialID,
    const rays::BSDFSample &bsdfSample,
    unsigned int &seed
) {
    if (bsdfSample.isDelta) { return 0.f; }

    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);
    const float xi3 = rnd(seed);

    const rays::LightSample lightSample = rays::Sampler::sampleDirectLights(
        intersection.point,
        intersection.frame,
        make_float3(xi1, xi2, xi3),
        params.lightIndices,
        params.lightIndexSize,
        params.triangles,
        params.environmentLight,
        *params.materialLookup
    );

    const rays::Ray shadowRay(
        intersection.point,
        lightSample.wi
    );

    PerRayData prd;
    prd.done = false;

    unsigned int p0, p1;
    rays::packPointer(&prd, p0, p1);
    optixTrace(
        params.handle,
        vec3_to_float3(shadowRay.origin()),
        vec3_to_float3(shadowRay.direction()),
        1e-4,
        lightSample.distance - 2e-4,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        1,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1
    );

    const bool isHit = !prd.done;
    if (isHit) {
        return rays::Vec3(0.f);
    } else {
        const rays::Vec3 wiLocal = intersection.frame.toLocal(lightSample.wi);
        const rays::Vec3 lightContribution = rays::Vec3(1.f)
            * lightSample.emitted
            * f(materialID, intersection.woLocal, wiLocal)
            * rays::WorldFrame::absCosTheta(intersection.normal, lightSample.wi)
            / lightSample.pdf;

        return lightContribution;
    }
}

__forceinline__ __device__ static rays::Vec3 direct(
    const rays::Intersection &intersection,
    const rays::BSDFSample &bsdfSample,
    const int &materialID,
    unsigned int &seed
) {
    rays::Vec3 result(0.f);

    result += directSampleLights(intersection, materialID, bsdfSample, seed);
    result += directSampleBSDF(intersection, bsdfSample, seed);

    return result;
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

    if (prd.done) {
        return params.environmentLight.getEmit(direction);
    } else {
        const rays::Intersection &intersection = prd.intersection;
        if (intersection.isFront()) {
            result += getEmit(prd.materialID);
        }
    }

    for (int path = 1; path < params.maxDepth; path++) {
        if (prd.done) { break; }

        const rays::Intersection &intersection = prd.intersection;
        const rays::BSDFSample bsdfSample = sample(prd.materialID, intersection, seed);

        result += beta * direct(intersection, bsdfSample, prd.materialID, seed);

        const rays::Frame &frame = intersection.frame;
        const rays::Vec3 bounceWorld = normalized(frame.toWorld(bsdfSample.wiLocal));

        beta *= bsdfSample.f
            * rays::TangentFrame::absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;

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
            0, 1, 0,
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

    if (prd.done) {
        return params.environmentLight.getEmit(direction);
    } else {
        const rays::Intersection &intersection = prd.intersection;
        if (intersection.isFront()) {
            result += getEmit(prd.materialID);
        }
    }

    for (int path = 1; path < params.maxDepth; path++) {
        if (prd.done) { break; }

        const rays::Intersection &intersection = prd.intersection;
        const rays::BSDFSample bsdfSample = sample(prd.materialID, intersection, seed);

        const rays::Frame &frame = intersection.frame;
        const rays::Vec3 bounceWorld = normalized(frame.toWorld(bsdfSample.wiLocal));

        const float3 normal = vec3_to_float3(intersection.normal);
        beta *= bsdfSample.f
            * rays::TangentFrame::absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;

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

        if (prd.done) {
            result += beta * params.environmentLight.getEmit(bounceWorld);
            break;
        }

        if (intersection.isFront()) {
            result += getEmit(prd.materialID) * beta;
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

    rays::Intersection intersection;
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    intersection.point = triangle.interpolate(u, v);
    intersection.normal = triangle.interpolateNormal(u, v);
    intersection.frame = rays::Frame(intersection.normal);
    intersection.woLocal = intersection.frame.toLocal(
        float3_to_vec3(-optixGetWorldRayDirection())
    );

    prd->intersection = intersection;
    prd->materialID = hitgroupData->materialID;
}
