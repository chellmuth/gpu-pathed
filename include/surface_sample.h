#pragma once

#include "core/vec3.h"
#include "world_frame.h"

namespace rays {

struct SurfaceSample {
    Vec3 point;
    Vec3 normal;
    float pdf;

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
