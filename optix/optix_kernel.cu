#include <optix.h>

#include "core/ray.h"
#include "core/vec3.h"
#include "frame.h"
#include "intersection.h"
#include "lights/sampler.h"
#include "materials/bsdf.h"
#include "materials/bsdf_sample.h"
#include "materials/lambertian.h"
#include "materials/types.h"
#include "math/mis.h"
#include "primitives/triangle.h"
#include "primitives/types.h"
#include "renderers/float3_helpers.h"
#include "renderers/optix.h"
#include "renderers/payload_helpers.h"
#include "renderers/random.h"
#include "tangent_frame.h"
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

__forceinline__ __device__ static float pdf(
    int materialID,
    const rays::Vec3 &wo,
    const rays::Vec3 &wi
) {
    rays::BSDF bsdf(params.materialLookup, materialID);
    return bsdf.pdf(wo, wi);
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
    if (bsdfSample.f.isZero()) { return rays::Vec3(0.f); }

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
        // fixme backside check?
        const rays::Vec3 emit = getEmit(prd.materialID);
        if (emit.isZero()) { return rays::Vec3(0.f); }

        const rays::Intersection bounceIntersection = prd.intersection;
        const float lightPDF = rays::Sampler::pdfSceneLights(
            intersection.point,
            bounceIntersection.point,
            bounceIntersection.normal,
            bounceIntersection.index,
            params.lightIndices,
            params.lightIndexSize,
            params.triangles,
            params.spheres,
            params.environmentLight
        );
        const float bsdfWeight = bsdfSample.isDelta
            ? 1.f
            : rays::MIS::balanceWeight(1, 1, bsdfSample.pdf, lightPDF);

        return emit
            * bsdfWeight
            * bsdfSample.f
            * rays::TangentFrame::absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;

    } else {
        const rays::Vec3 emit = params.environmentLight.getEmit(bounceRay.direction());
        if (emit.isZero()) { return rays::Vec3(0.f); }

        const float lightPDF = rays::Sampler::pdfEnvironmentLight(
            bounceRay.direction(),
            params.environmentLight,
            params.lightIndexSize
        );

        const float bsdfWeight = bsdfSample.isDelta
            ? 1.f
            : rays::MIS::balanceWeight(1, 1, bsdfSample.pdf, lightPDF);

        return emit
            * bsdfWeight
            * bsdfSample.f
            * rays::TangentFrame::absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;

    }
}

__forceinline__ __device__ static rays::Vec3 directSampleLights(
    const rays::Intersection &intersection,
    int materialID,
    const rays::BSDFSample &bsdfSample,
    unsigned int &seed
) {
    if (bsdfSample.isDelta) { return rays::Vec3(0.f); }

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
        params.spheres,
        params.environmentLight,
        *params.materialLookup
    );

    if (dot(lightSample.normal, lightSample.wi) >= 0.f) {
        return rays::Vec3(0.f);
    }

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
        0, 1, 0, // sbt fields
        p0, p1
    );

    const bool isHit = !prd.done;
    if (isHit) {
        return rays::Vec3(0.f);
    }

    const rays::Vec3 wiLocal = intersection.frame.toLocal(lightSample.wi);
    const float bsdfPDF = pdf(bsdfSample.materialID, intersection.woLocal, wiLocal);
    const float lightWeight = rays::MIS::balanceWeight(1, 1, lightSample.pdf, bsdfPDF);

    const rays::Vec3 lightContribution = rays::Vec3(1.f)
        * lightSample.emitted
        * lightWeight
        * f(materialID, intersection.woLocal, wiLocal)
        * rays::WorldFrame::absCosTheta(intersection.normal, lightSample.wi)
        / lightSample.pdf;

    return lightContribution;
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
    rays::Intersection intersection;

    if (optixIsTriangleHit()) {
        const rays::Triangle &triangle = params.triangles[primitiveIndex];

        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;
        intersection.point = triangle.interpolate(u, v);
        intersection.normal = triangle.interpolateNormal(u, v);
        intersection.frame = rays::Frame(intersection.normal);
        intersection.woLocal = intersection.frame.toLocal(
            float3_to_vec3(-optixGetWorldRayDirection())
        );
        intersection.index = rays::PrimitiveIndex{
            rays::PrimitiveType::Triangle,
            primitiveIndex
        };
    } else {
        const rays::Vec3 point(
            int_as_float(optixGetAttribute_0()),
            int_as_float(optixGetAttribute_1()),
            int_as_float(optixGetAttribute_2())
        );

        const rays::Vec3 normal(
            int_as_float(optixGetAttribute_3()),
            int_as_float(optixGetAttribute_4()),
            int_as_float(optixGetAttribute_5())
        );

        intersection.point = point;
        intersection.normal = normalized(normal);
        intersection.frame = rays::Frame(intersection.normal);
        intersection.woLocal = intersection.frame.toLocal(
            float3_to_vec3(-optixGetWorldRayDirection())
        );
        intersection.index = rays::PrimitiveIndex{
            rays::PrimitiveType::Sphere,
            primitiveIndex
        };
    }

    rays::HitGroupData* hitgroupData = reinterpret_cast<rays::HitGroupData *>(optixGetSbtDataPointer());
    prd->intersection = intersection;
    prd->materialID = hitgroupData->materialID;
}

extern "C" __global__ void __intersection__sphere()
{
    const float tMax = optixGetRayTmax();
    const float tMin = optixGetRayTmin();

    const rays::Ray ray(
        float3_to_vec3(optixGetObjectRayOrigin()),
        float3_to_vec3(optixGetObjectRayDirection())
    );

    const int primitiveIndex = optixGetPrimitiveIndex();
    const rays::Sphere &sphere = params.spheres[primitiveIndex];
    const rays::Vec3 center = sphere.getCenter();
    const float radius = sphere.getRadius();

    const rays::Vec3 oc = ray.origin() - center;
    const float a = dot(ray.direction(), ray.direction());
    const float b = dot(oc, ray.direction());
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < tMax && temp > tMin) {
            const rays::Vec3 point = ray.at(temp);
            const rays::Vec3 normal = normalized((point - center) / radius);
            const rays::Frame f(normal);

            unsigned int p0, p1, p2;
            p0 = float_as_int( point.x() );
            p1 = float_as_int( point.y() );
            p2 = float_as_int( point.z() );

            unsigned int n0, n1, n2;
            n0 = float_as_int( normal.x() );
            n1 = float_as_int( normal.y() );
            n2 = float_as_int( normal.z() );

            optixReportIntersection(
                temp,       // t hit
                0,          // user hit kind
                p0, p1, p2,
                n0, n1, n2
            );
        }
    }
}
