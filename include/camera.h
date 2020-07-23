#pragma once

#include <curand_kernel.h>

#include "ray.h"
#include "vec3.h"

namespace rays {

struct Resolution {
    int x;
    int y;
};

class Camera {
public:
    __host__ __device__ Camera(
        const Vec3 &origin,
        const Vec3 &target,
        float verticalFOV,
        const Resolution &resolution
    );

    __device__ Ray generateRay(int row, int col, curandState &randState) const;
    __device__ Ray generateRay(int row, int col) const;

    float getVerticalFOV() const { return m_verticalFOV; }
    Resolution getResolution() const { return m_resolution; }

    Vec3 getOrigin() const { return m_origin; }
    Vec3 getTarget() const { return m_target; }
    Vec3 getUp() const { return Vec3(0.f, 1.f, 0.f); }

private:
    __device__ Ray generateRay(int row, int col, float2 samples) const;

    Vec3 m_origin;
    Vec3 m_target;
    float m_verticalFOV;
    Resolution m_resolution;
};

}
