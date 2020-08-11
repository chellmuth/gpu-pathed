#pragma once

#include "core/vec3.h"

namespace rays { namespace Color {

__device__ inline Vec3 toSRGB(const Vec3 &linear)
{
    const float gamma = 2.2;

    const float r = linear.r();
    const float g = linear.g();
    const float b = linear.b();

    const float sR = r <= 0.0031308f
        ? r * 12.92f
        : 1.055f * std::pow(r, (1.f / gamma)) - 0.055f;

    const float sG = g <= 0.0031308f
        ? g * 12.92f
        : 1.055f * std::pow(g, (1.f / gamma)) - 0.055f;

    const float sB = b <= 0.0031308f
        ? b * 12.92f
        : 1.055f * std::pow(b, (1.f / gamma)) - 0.055f;

    return Vec3(sR, sG, sB);
}

} }

