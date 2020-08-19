#pragma once

#include <cuda_runtime.h>

#include "core/ray.h"
#include "core/vec3.h"
#include "frame.h"
#include "lights/environment_light.h"
#include "materials/material_lookup.h"
#include "primitives/triangle.h"
#include "surface_sample.h"
#include "world_frame.h"

namespace rays {

struct LightSample {
    Vec3 wi;
    float distance;
    float pdf;
    Vec3 emitted;
};

}

namespace rays { namespace Sampler {

__device__ inline LightSample sampleSceneLights(
    int lightChoice,
    float choicePDF,
    const Vec3 &hitPoint,
    const float3 &samples,
    const int *lightIndices,
    const Triangle *triangles,
    const MaterialLookup &materialLookup
) {
    const Triangle &triangle = triangles[lightIndices[lightChoice]];
    const SurfaceSample sample = triangle.sample(samples.y, samples.z);

    const Vec3 direction = sample.point - hitPoint;
    const float distance = direction.length();
    const float pdf = sample.solidAnglePDF(hitPoint) * choicePDF;

    const LightSample lightSample = {
        .wi = normalized(direction),
        .distance = distance,
        .pdf = pdf,
        .emitted = materialLookup.getEmit(triangle.materialID())
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
    const int *lightIndices,
    int lightIndexSize,
    const Triangle *triangles,
    const EnvironmentLight &environmentLight,
    const MaterialLookup &materialLookup
) {
    int choiceSize = lightIndexSize;
    if (environmentLight.getType() != EnvironmentLightType::None) {
        choiceSize += 1;
    }

    const int lightChoice = (int)floorf(samples.x * choiceSize);
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
