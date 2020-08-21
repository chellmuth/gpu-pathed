#include "world.h"

namespace rays {

__device__ bool World::hit(
    const Ray& ray,
    float tMin,
    float tMax,
    HitRecord& record
) const {
    bool isHit = false;
    double tNear = tMax;

    for (int i = 0; i < m_triangleSize; i++) {
        HitRecord tempRecord;
        if (m_triangles[i].hit(ray, tMin, tNear, tempRecord)) {
            isHit = true;
            tNear = tempRecord.t;
            record = tempRecord;
            record.index = {
                PrimitiveType::Triangle,
                i
            };
        }
    }

    for (int i = 0; i < m_sphereSize; i++) {
        HitRecord tempRecord;
        if (m_spheres[i].hit(ray, tMin, tNear, tempRecord)) {
            isHit = true;
            tNear = tempRecord.t;
            record = tempRecord;
            record.index = {
                PrimitiveType::Sphere,
                i
            };
        }
    }

    return isHit;
}

}
