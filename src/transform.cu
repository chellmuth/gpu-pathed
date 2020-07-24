#include "transform.h"

#include <iostream>

namespace rays {

static constexpr float identity[4][4] = {
    1.f, 0.f, 0.f, 0.f,
    0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 1.f,
};

Transform::Transform()
    : Transform(identity)
{}

Transform::Transform(const float matrix[4][4])
{
    for (int row = 0; row < 4; row++ ) {
        for (int col = 0; col < 4; col++ ) {
            m_matrix[row][col] = matrix[row][col];
        }
    }
}

__device__ Vec3 Transform::applyPoint(const Vec3 &point) const
{
    const float x = point.x();
    const float y = point.y();
    const float z = point.z();

    return Vec3(
        m_matrix[0][0] * x + m_matrix[0][1] * y + m_matrix[0][2] * z + m_matrix[0][3],
        m_matrix[1][0] * x + m_matrix[1][1] * y + m_matrix[1][2] * z + m_matrix[1][3],
        m_matrix[2][0] * x + m_matrix[2][1] * y + m_matrix[2][2] * z + m_matrix[2][3]
    );
}

__device__ Vec3 Transform::applyVector(const Vec3 &vector) const
{
    const float x = vector.x();
    const float y = vector.y();
    const float z = vector.z();

    return Vec3(
        m_matrix[0][0] * x + m_matrix[0][1] * y + m_matrix[0][2] * z,
        m_matrix[1][0] * x + m_matrix[1][1] * y + m_matrix[1][2] * z,
        m_matrix[2][0] * x + m_matrix[2][1] * y + m_matrix[2][2] * z
    );
}

__device__ Ray Transform::apply(const Ray &ray) const
{
    return Ray(
        applyPoint(ray.origin()),
        applyVector(ray.direction())
    );
}

Transform lookAt(
    const Vec3 &source,
    const Vec3 &target,
    const Vec3 &up
) {
    const Vec3 direction = normalized(source - target);

    if (direction == up) {
        std::cerr << "Look direction cannot equal up vector - quitting!" << std::endl;
        exit(1);
    }

    const Vec3 xAxis = normalized(cross(normalized(up), direction));
    const Vec3 yAxis = normalized(cross(direction, xAxis));

    float matrix[4][4] {
        { xAxis.x(), yAxis.x(), direction.x(), source.x() },
        { xAxis.y(), yAxis.y(), direction.y(), source.y() },
        { xAxis.z(), yAxis.z(), direction.z(), source.z() },
        { 0.f, 0.f, 0.f, 1.f }
    };

    return Transform(matrix);
}

}
