#pragma once

namespace rays { namespace util {

__device__ inline float clamp(float value, float lowest, float highest)
{
    return fminf(highest, fmaxf(value, lowest));
}

__device__ inline float square(float x) {
    return x * x;
}


} }
