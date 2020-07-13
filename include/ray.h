#pragma once

#include "vec3.h"

namespace rays {

class Ray
{
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction)
        : m_origin(origin),
          m_direction(direction)
    {}

    __host__ __device__ Vec3 origin() const { return m_origin; }
    __host__ __device__ Vec3 direction() const { return m_direction; }
    __host__ __device__ Vec3 at(float t) const { return m_origin + t * m_direction; }

private:
    Vec3 m_origin;
    Vec3 m_direction;
};

}
