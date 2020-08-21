#pragma once

#include "materials/lambertian.h"
#include "core/sample.h"
#include "core/ray.h"
#include "core/vec3.h"
#include "surface_sample.h"

namespace rays {

class Sphere  {
public:
    __host__ __device__ Sphere(
        const Vec3 &center,
        float radius,
        int materialID
    )
        : m_center(center),
          m_radius(radius),
          m_materialID(materialID)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

    __device__ SurfaceSample sample(float xi1, float xi2) const {
        const Vec3 pointLocal = Sample::uniformSphere(xi1, xi2);
        const Vec3 pointWorld = pointLocal * m_radius + m_center;

        SurfaceSample sample = {
            .point = pointWorld,
            .normal = pointLocal,
            .pdf = pdfArea()
        };
        return sample;
    }

    __device__ float pdfArea() const {
        return 1.f / area();
    }

    __host__ __device__ int materialID() const { return m_materialID; }

    __host__ __device__ Vec3 getCenter() const {
        return m_center;
    }

    __host__ __device__ float getRadius() const {
        return m_radius;
    }

private:
    __device__ float area() const {
        return 4.f * M_PI * m_radius * m_radius;
    }

    Vec3 m_center;
    float m_radius;
    int m_materialID;
};

}
