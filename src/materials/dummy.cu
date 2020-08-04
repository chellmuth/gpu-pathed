#include "materials/dummy.h"

namespace rays {

__device__ Vec3 Dummy::getEmit(const HitRecord &hit) const
{
    return Vec3(0.f);
}

__host__ void Dummy::setEmit(const Vec3 &emit)
{
}

__host__ void Dummy::setAlbedo(const Vec3 &albedo)
{
}


}
