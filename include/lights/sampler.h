#pragma once

#include <cuda_runtime.h>

#include "core/ray.h"
#include "core/vec3.h"
#include "frame.h"
#include "lights/environment_light.h"
#include "lights/types.h"
#include "materials/material_lookup.h"
#include "primitives/triangle.h"
#include "surface_sample.h"
#include "world_frame.h"

namespace rays { namespace Sampler {

__device__ inline LightSample sampleSceneLights(
    int lightChoice,
    float choicePDF,
    const Vec3 &hitPoint,
    const float3 &samples,
    const LightIndex *lightIndices,
    const Triangle *triangles,
    const Sphere *spheres,
    const MaterialLookup &materialLookup
) {
    const LightIndex lightIndex = lightIndices[lightChoice];

    SurfaceSample sample;
    int materialID;
    switch (lightIndex.primitiveType) {
    case PrimitiveType::Triangle: {
        const Triangle &triangle = triangles[lightIndex.index];
        sample = triangle.sample(samples.y, samples.z);
        materialID = triangle.materialID();
        break;
    }
    case PrimitiveType::Sphere: {
        const Sphere &sphere = spheres[lightIndex.index];
        sample = sphere.sample(samples.y, samples.z);
        materialID = sphere.materialID();
        break;
    }
    }

    const Vec3 direction = sample.point - hitPoint;
    const float distance = direction.length();
    const float pdf = sample.solidAnglePDF(hitPoint) * choicePDF;

    const LightSample lightSample = {
        .wi = normalized(direction),
        .distance = distance,
        .pdf = pdf,
        .emitted = materialLookup.getEmit(materialID)
    };
    return lightSample;
}

__device__ inline LightSample sampleEnvironmentLight(
    float choicePDF,
    const Vec3 &hitPoint,
    const Frame &frame,
    const float3 &samples,
    const EnvironmentLight &environmentLight
) {
    const EnvironmentLightSample sample = environmentLight.sample(
        hitPoint,
        frame,
        samples.y,
        samples.z
    );

    const LightSample lightSample = {
        .wi = normalized(sample.occlusionRay.direction()),
        .distance = 1e16, // fixme
        .pdf = sample.pdf * choicePDF,
        .emitted = sample.emitted
    };
    return lightSample;
}

__device__ inline LightSample sampleDirectLights(
    const Vec3 &hitPoint,
    const Frame &frame,
    const float3 &samples,
    const LightIndex *lightIndices,
    int lightIndexSize,
    const Triangle *triangles,
    const Sphere *spheres,
    const EnvironmentLight &environmentLight,
    const MaterialLookup &materialLookup
) {
    int choiceSize = lightIndexSize;
    if (environmentLight.getType() != EnvironmentLightType::None) {
        choiceSize += 1;
    }

    // todo: modify curand code to be [0, 1) instead of [0, 1)
    const int lightChoice = fminf(choiceSize - 1, (int)floorf(samples.x * choiceSize));
    const float choicePDF = 1.f / choiceSize;

    // Area lights or environment light
    if (lightChoice < lightIndexSize) {
        return sampleSceneLights(
            lightChoice,
            choicePDF,
            hitPoint,
            samples,
            lightIndices,
            triangles,
            spheres,
            materialLookup
        );
    } else {
        return sampleEnvironmentLight(
            choicePDF,
            hitPoint,
            frame,
            samples,
            environmentLight
        );
    }
}

} };
