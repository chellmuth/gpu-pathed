#include "primitive.h"

namespace rays {

__device__ bool PrimitiveList::hit(
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
