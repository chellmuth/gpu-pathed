#include "materials/mirror.h"

namespace rays {

__device__ Vec3 Mirror::getEmit(const HitRecord &hit) const
{
    return Vec3(0.f);
}

__host__ void Mirror::setEmit(const Vec3 &emit)
{
}

__host__ void Mirror::setAlbedo(const Vec3 &albedo)
{
}


}
