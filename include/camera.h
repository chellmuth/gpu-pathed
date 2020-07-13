#pragma once

#include <curand_kernel.h>

#include "ray.h"
#include "vec3.h"

namespace rays {

struct Resolution {
    int x;
    int y;
};

class Camera {
public:
    __device__ Camera(
        const Vec3 &origin,
        float verticalFOV,
        const Resolution &resolution
    );
    __device__ Ray generateRay(int row, int col, curandState &randState) const;

private:
    Vec3 m_origin;
    float m_verticalFOV;
    Resolution m_resolution;
};

}
