#include "triangle.h"

namespace rays {

__device__ bool Triangle::hit(
    const Ray& ray,
    float tMin,
    float tMax,
    HitRecord& record
) const {
    const Vec3 e1 = m_p1 - m_p0;
    const Vec3 e2 = m_p2 - m_p0;

    const Vec3 s1 = cross(ray.direction(), e2);
    const float divisor = dot(s1, e1);

    if (divisor == 0.f) { return false; }

    const float inverseDivisor = 1.f / divisor;

    const Vec3 s = ray.origin() - m_p0;
    const float b1 = dot(s, s1) * inverseDivisor;

    if (b1 < 0.f || b1 > 1.f) { return false; }

    const Vec3 s2 = cross(s, e1);
    const float b2 = dot(ray.direction(), s2) * inverseDivisor;

    if (b2 < 0.f || (b1 + b2) > 1.f) { return false; }

    const float t = dot(e2, s2) * inverseDivisor;
    if (t < tMin || t > tMax) { return false; }

    const Vec3 hitPoint = m_p0 * (1.f - b1 - b2) + (m_p1 * b1) + (m_p2 * b2);

    const Vec3 normal = normalized(cross(e1, e2));
    const Frame frame(normal);

    record.t = t;
    record.point = hitPoint;
    record.normal = normal;
    record.wo = normalized(frame.toLocal(-ray.direction()));
    record.materialIndex = m_materialIndex;

    return true;
}

__device__ float Triangle::area() const
{
    const Vec3 e1 = m_p1 - m_p0;
    const Vec3 e2 = m_p2 - m_p0;

    const Vec3 crossed = cross(e1, e2);
    return fabsf(crossed.length() / 2.f);
}

__device__ SurfaceSample Triangle::sample(curandState &randState) const
{
    const float r1 = curand_uniform(&randState);
    const float r2 = curand_uniform(&randState);

    const float a = 1 - sqrt(r1);
    const float b = sqrt(r1) * (1 - r2);
    const float c = 1 - a - b;

    const Vec3 point = m_p0 * a + m_p1 * b + m_p2 * c;

    const Vec3 e1 = m_p1 - m_p0;
    const Vec3 e2 = m_p2 - m_p0;
    const Vec3 normal = normalized(cross(e1, e2));

    SurfaceSample sample = {
        .point = point,
        .normal = normal,
        .pdf = 1.f / area(),
    };
    return sample;
}

}
