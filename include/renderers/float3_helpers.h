#pragma once

#include "renderers/random.h"
#include "vec3.h"


__forceinline__ __device__ float2 sample_float2(unsigned int &seed)
{
    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);
    return make_float2(xi1, xi2);
}

__forceinline__ __device__ float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __device__ void operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

__forceinline__ __device__ void operator*=(float3& a, const float3& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__forceinline__ __device__ float3 operator*(const float s, const float3 &a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __device__ float3 operator*(const float3 &a, const float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float3 operator/(const float3 &a, const float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__forceinline__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float3 make_float3(const float f)
{
    return make_float3(f, f, f);
}

__forceinline__ __device__ float3 vec3_to_float3(const rays::Vec3 &vec3)
{
    return make_float3(
        vec3.x(),
        vec3.y(),
        vec3.z()
    );
}

__forceinline__ __device__ rays::Vec3 float3_to_vec3(const float3 &f3)
{
    return rays::Vec3(
        f3.x,
        f3.y,
        f3.z
    );
}
