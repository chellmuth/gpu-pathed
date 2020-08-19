#pragma once

#include <assert.h>
#include <string>

#include "frame.h"
#include "math/distribution.h"
#include "math/spherical_coordinates.h"
#include "core/ray.h"
#include "core/sample.h"
#include "core/vec3.h"

namespace rays {

struct EnvironmentLightSample {
    Ray occlusionRay;
    float pdf;
    Vec3 emitted;
};

enum class EnvironmentLightType {
    None,
    Image
};

class EnvironmentLight {
public:
    __device__ EnvironmentLight() {}

    static EnvironmentLight None() {
        return EnvironmentLight(EnvironmentLightType::None);
    }

    EnvironmentLight(float *data, PhiThetaDistribution distribution, int width, int height)
        : m_type(EnvironmentLightType::Image),
          m_data(data),
          m_distribution(distribution),
          m_width(width),
          m_height(height)
    {}

    __device__ EnvironmentLightType getType() const {
        return m_type;
    }

    __device__ EnvironmentLightSample sample(
        const Vec3 &point,
        const Frame &frame,
        float xi1,
        float xi2
    ) const {
        float pdf;
        float2 xis{xi1, xi2};
        const Vec3 wi = m_distribution.sample(&pdf, xis);

        const EnvironmentLightSample sample = {
            .occlusionRay = Ray(point, wi),
            .pdf = pdf,
            .emitted = getEmit(wi)
        };
        return sample;
    }

    __device__ Vec3 getEmit(const Vec3 &wi) const {
        if (m_type == EnvironmentLightType::None) { return Vec3(0.f); }

        float phi, theta;
        Coordinates::cartesianToSpherical(wi, &phi, &theta);

        const int x = floorf(phi / (M_PI * 2.f) * m_width);
        const int y = floorf(theta / M_PI * m_height);

        const int offset = (4 * m_width) * y + 4 * x;
        return Vec3(
            m_data[offset + 0],
            m_data[offset + 1],
            m_data[offset + 2]
        );
    }

private:
    EnvironmentLight(EnvironmentLightType type)
        : m_type(type)
    {
        assert(type == EnvironmentLightType::None);
    }


    EnvironmentLightType m_type;

    float *m_data;
    PhiThetaDistribution m_distribution;
    int m_width;
    int m_height;
};


class EnvironmentLightParams {
public:
    EnvironmentLightParams()
        : m_type(EnvironmentLightType::None)
    {}

    EnvironmentLightParams(const std::string &filename)
        : m_type(EnvironmentLightType::Image),
          m_filename(filename)
    {}

    EnvironmentLight createEnvironmentLight() const;

private:
    EnvironmentLightType m_type;
    std::string m_filename;
};

}
