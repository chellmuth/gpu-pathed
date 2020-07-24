#pragma once

#include <curand_kernel.h>

#include "ray.h"
#include "transform.h"
#include "vec3.h"

namespace rays {

struct Resolution {
    int x;
    int y;
};

class Camera {
public:
    Camera(
        const Vec3 &origin,
        const Vec3 &target,
        const Vec3 &up,
        float verticalFOV,
        const Resolution &resolution
    );

    __device__ Ray generateRay(int row, int col, curandState &randState) const;
    __device__ Ray generateRay(int row, int col) const;

    float getVerticalFOV() const { return m_verticalFOV; }
    Resolution getResolution() const { return m_resolution; }

    Vec3 getOrigin() const { return m_origin; }
    Vec3 getTarget() const { return m_target; }
    Vec3 getUp() const { return m_up; }

private:
    __device__ Ray generateRay(int row, int col, float2 samples) const;

    Transform m_cameraToWorld;

    Vec3 m_origin;
    Vec3 m_target;
    Vec3 m_up;
    float m_verticalFOV;
    Resolution m_resolution;
};

}
