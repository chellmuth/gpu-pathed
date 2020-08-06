#include "materials/glass.h"

namespace rays {

__device__ Vec3 Glass::getEmit(const HitRecord &hit) const
{
    return Vec3(0.f);
}

__host__ void Glass::setEmit(const Vec3 &emit)
{
}

__host__ void Glass::setAlbedo(const Vec3 &albedo)
{
}

}
