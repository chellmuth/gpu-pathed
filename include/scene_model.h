#pragma once

#include <functional>
#include <iostream>

#include "material.h"
#include "path_tracer.h"
#include "vec3.h"

namespace rays {

class PathTracer;

class SceneModel {
public:
    SceneModel(PathTracer *pathTracer, const Vec3 &color, float lightPosition);

    SceneModel(const SceneModel &other) = delete;
    SceneModel(SceneModel&& other) = delete;

    void subscribe(std::function<void()> callback);

    void setColor(float r, float g, float b);
    Vec3 getColor() const;

    int getMaterialIndex() const;
    void setMaterialIndex(int materialIndex, const Vec3 &albedo);

    void setLightPosition(float lightPosition);
    float getLightPosition() const;

    int getSpp() const;

private:
    PathTracer *m_pathTracer;

    int m_materialIndex;

    int m_spp;
    float m_r, m_g, m_b;
    float m_lightPosition;

    std::function<void()> m_callback;
};

}
