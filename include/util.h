#pragma once

namespace rays { namespace util {

__device__ inline float clamp(float value, float lowest, float highest)
{
    return fminf(highest, fmaxf(value, lowest));
}

__device__ inline float square(float x) {
    return x * x;
}

__device__ inline float sign(float x) {
    return x < 0.f
        ? -1.f
        : 1.f;
}

} }
