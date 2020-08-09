#pragma once

#include "vec3.h"

namespace rays {

enum class EnvironmentLightType {
    None,
    Constant
};

class EnvironmentLight {
public:
    EnvironmentLight()
        : m_type(EnvironmentLightType::None),
          m_color(Vec3(0.f))
    {}

    EnvironmentLight(const Vec3 color)
        : m_type(EnvironmentLightType::Constant),
          m_color(color)
    {}

    __device__ Vec3 getEmit(const Vec3 &wi) const {
        switch (m_type) {
        case EnvironmentLightType::None: {
            return Vec3(0.f);
        }
        case EnvironmentLightType::Constant: {
            return m_color;
        }
        }
        return Vec3(0.f);
    }

private:
    EnvironmentLightType m_type;
    Vec3 m_color;
};

}
