#include "sphere.h"

#include "frame.h"

namespace rays {

__device__ bool Sphere::hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const
{
    const Vec3 oc = ray.origin() - m_center;
    const float a = dot(ray.direction(), ray.direction());
    const float b = dot(oc, ray.direction());
    const float c = dot(oc, oc) - m_radius * m_radius;
    const float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < tMax && temp > tMin) {
            const Vec3 point = ray.at(temp);
            const Vec3 normal = normalized((point - m_center) / m_radius);
            const Frame f(normal);

            record.t = temp;
            record.point = point;
            record.normal = normal;
            record.wo = normalized(f.toLocal(-ray.direction()));
            record.materialPtr = m_materialPtr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < tMax && temp > tMin) {
            const Vec3 point = ray.at(temp);
            const Vec3 normal = normalized((point - m_center) / m_radius);
            const Frame f(normal);

            record.t = temp;
            record.point = point;
            record.normal = normal;
            record.wo = normalized(f.toLocal(-ray.direction()));
            record.materialPtr = m_materialPtr;
            return true;
        }
    }
    return false;
}

}
