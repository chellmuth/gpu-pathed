#pragma once

#include <cuda_runtime.h>

#include "materials/material_table.h"
#include "surface_sample.h"
#include "triangle.h"
#include "vec3.h"
#include "world_frame.h"

namespace rays {

struct LightSample {
    Vec3 point;
    Vec3 normal;
    float pdf;
    MaterialIndex materialIndex;

    __device__ float solidAnglePDF(const Vec3 &referencePoint) const
    {
        const Vec3 lightDirection = point - referencePoint;
        const Vec3 lightWo = -normalized(lightDirection);
        const float distance = lightDirection.length();

        const float distance2 = distance * distance;
        const float projectedArea = WorldFrame::cosTheta(normal, lightWo);

        return pdf * distance2 / projectedArea;
    }

};

}

namespace rays { namespace Sampler {

    __device__ inline LightSample sampleDirectLights(
        Vec3 hitPoint,
        const float3 &samples,
        const int *lightIndices,
        int lightIndexSize,
        const Triangle *triangles
    ) {
        const int lightChoice = (int)floorf(samples.x * lightIndexSize);
        const float choicePDF = 1.f / lightIndexSize;

        const Triangle &triangle = triangles[lightIndices[lightChoice]];
        const SurfaceSample sample = triangle.sample(samples.y, samples.z);

        LightSample lightSample = {
            sample.point,
            sample.normal,
            sample.pdf * choicePDF,
            triangle.materialIndex()
        };
        return lightSample;
    }


} };
