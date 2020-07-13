#include "camera.h"

#include <cmath>

#include "vec3.h"

namespace rays {

__device__ Camera::Camera(const Vec3 &origin, float verticalFOV, const Resolution &resolution)
    : m_origin(origin),
      m_verticalFOV(verticalFOV),
      m_resolution(resolution)
{}

__device__ Ray Camera::generateRay(int row, int col, curandState &randState) const
{
    const float top = std::tan(m_verticalFOV / 2.f);
    const float height = top * 2.f;

    const float aspectRatio = 1.f * m_resolution.x / m_resolution.y;
    const float width = height * aspectRatio;
    const float right = width / 2.f;

    const float xCanonical = (col + curand_uniform(&randState)) / m_resolution.x;
    const float yCanonical = (row + curand_uniform(&randState)) / m_resolution.y;

    const float y = yCanonical * height - top;
    const float x = xCanonical * width - right;

    const Vec3 direction = normalized(Vec3(x, y, -1));

    return Ray(m_origin, direction);
}

}
