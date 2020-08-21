#pragma once

#include "core/vec3.h"
#include "world_frame.h"

namespace rays { namespace Measure {

__device__ inline float areaToSolidAngle(
    float areaPDF,
    const Vec3 &referencePoint,
    const Vec3 &surfacePoint,
    const Vec3 &surfaceNormal
) {
    const Vec3 surfaceDirection = referencePoint - surfacePoint;
    const Vec3 surfaceWo = normalized(surfaceDirection);
    const float distance = surfaceDirection.length();

    const float distance2 = distance * distance;
    const float projectedArea = WorldFrame::cosTheta(surfaceNormal, surfaceWo);

    const float solidAnglePDF = areaPDF * distance2 / projectedArea;
    return solidAnglePDF;
}


} }
