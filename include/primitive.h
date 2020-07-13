#pragma once

#include "ray.h"
#include "vec3.h"

namespace rays {

class Material;

struct HitRecord {
    float t;
    Vec3 point;
    Vec3 normal;
    Vec3 wo;
    Material *materialPtr;

    __device__ bool isFront() const {
        return wo.z() >= 0.f;
    }
};

class Primitive {
public:
    __device__ virtual bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord &record
    ) const = 0;
};

class PrimitiveList {
public:
    __device__ PrimitiveList(Primitive **list, int size)
        : m_list(list), m_size(size) {}

    __device__ virtual bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

private:
    Primitive **m_list;
    int m_size;
};

__device__ inline bool PrimitiveList::hit(
    const Ray& ray,
    float tMin,
    float tMax,
    HitRecord& record
) const {
    bool isHit = false;
    double tNear = tMax;

    for (int i = 0; i < m_size; i++) {
        HitRecord tempRecord;
        if (m_list[i]->hit(ray, tMin, tNear, tempRecord)) {
            isHit = true;
            tNear = tempRecord.t;
            record = tempRecord;
        }
    }

    return isHit;
}

}
