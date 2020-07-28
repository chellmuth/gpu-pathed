#include "camera.h"

#include <cmath>

namespace rays {

Camera::Camera(
    const Vec3 &origin,
    const Vec3 &target,
    const Vec3 &up,
    float verticalFOV,
    const Resolution &resolution
) : m_origin(origin),
    m_target(target),
    m_up(up),
    m_verticalFOV(verticalFOV),
    m_resolution(resolution)
{
    m_cameraToWorld = lookAt(m_origin, m_target, m_up);
}

__device__ Ray Camera::generateRay(int row, int col) const
{
    return generateRay(row, col, make_float2(0.5f, 0.5f));
}

__device__ Ray Camera::generateRay(int row, int col, curandState &randState) const
{
    const float xi1 = curand_uniform(&randState);
    const float xi2 = curand_uniform(&randState);
    return generateRay(row, col, make_float2(xi1, xi2));
}

}
