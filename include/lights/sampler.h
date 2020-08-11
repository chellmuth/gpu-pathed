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
    Ray occlusionRay;
    Vec3 wi;
    float distance;
    float pdf;
    Vec3 emitted;
};

}

namespace rays { namespace Sampler {

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
    const int lightChoice = (int)floorf(samples.x * (lightIndexSize + 1));
    const float choicePDF = 1.f / (lightIndexSize + 1);

    // Area lights or environment light
    if (lightChoice < lightIndexSize) {
        const Triangle &triangle = triangles[lightIndices[lightChoice]];
        const SurfaceSample sample = triangle.sample(samples.y, samples.z);

        const Vec3 direction = sample.point - hitPoint;
        const float distance = direction.length();
        const float pdf = sample.solidAnglePDF(hitPoint) * choicePDF;

        const LightSample lightSample = {
            .occlusionRay = Ray(sample.point, normalized(direction)),
            .wi = normalized(direction),
            .distance = distance,
            .pdf = pdf,
            .emitted = materialLookup.getEmit(triangle.materialID())
        };
        return lightSample;
    } else {
        const EnvironmentLightSample sample = environmentLight.sample(
            hitPoint,
            frame,
            samples.y,
            samples.z
        );

        const LightSample lightSample = {
            .occlusionRay = Ray(Vec3(0.f), Vec3(0.f)), // fixme
            .wi = normalized(sample.occlusionRay.direction()),
            .distance = 1e16, // fixme
            .pdf = sample.pdf * choicePDF,
            .emitted = sample.emitted
        };
        return lightSample;
    }
}

} };
