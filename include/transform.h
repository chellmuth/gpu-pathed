#pragma once

#include "ray.h"
#include "vec3.h"

namespace rays {

class Transform {
public:
    Transform();
    Transform(const float matrix[4][4]);

    __device__ Vec3 applyPoint(const Vec3 &point) const;
    __device__ Vec3 applyVector(const Vec3 &vector) const;
    __device__ Ray apply(const Ray &ray) const;

private:
    float m_matrix[4][4];
};

Transform lookAt(const Vec3 &source, const Vec3 &target, const Vec3 &up);

}
