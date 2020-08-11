#pragma once

#include <string>

#include "math/spherical_coordinates.h"
#include "core/vec3.h"

namespace rays {

class EnvironmentLightParams {
public:
    EnvironmentLightParams() {}

    EnvironmentLightParams(const std::string &filename)
        : m_filename(filename)
    {}

private:
    std::string m_filename;
};

class EnvironmentLight {
public:
    EnvironmentLight() {}

    EnvironmentLight(float *data, int width, int height)
        : m_data(data),
          m_width(width),
          m_height(height)
    {}

    __device__ Vec3 getEmit(const Vec3 &wi) const {
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
    float *m_data;
    int m_width;
    int m_height;
};

}
