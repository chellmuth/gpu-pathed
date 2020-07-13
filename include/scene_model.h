#pragma once

#include <functional>
#include <iostream>

#include "vec3.h"

namespace rays {

class SceneModel {
public:
    SceneModel(const Vec3 &color, float lightPosition)
        : m_r(color.r()),
          m_g(color.g()),
          m_b(color.b()),
          m_lightPosition(lightPosition),
          m_spp(0)
    {}

    SceneModel(const SceneModel &other) = delete;
    SceneModel(SceneModel&& other) = delete;

    void subscribe(std::function<void()> callback) {
        m_callback = callback;
    }

    void setColor(float r, float g, float b) {
        m_r = r;
        m_g = g;
        m_b = b;

        m_callback();
    }
    Vec3 getColor() { return Vec3(m_r, m_g, m_b); }

    void setLightPosition(float lightPosition) {
        m_lightPosition = lightPosition;
        m_callback();
    }
    float getLightPosition() { return m_lightPosition; }

    void updateSpp(int spp) { m_spp = spp; }
    int getSpp() { return m_spp; }

private:
    int m_spp;
    float m_r, m_g, m_b;
    float m_lightPosition;

    std::function<void()> m_callback;
};

}
