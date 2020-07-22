#pragma once

#include <functional>
#include <iostream>

#include "material.h"
#include "path_tracer.h"
#include "scene.h"
#include "vec3.h"

namespace rays {

class PathTracer;

class SceneModel {
public:
    SceneModel(
        const PathTracer *pathTracer,
        const Scene *scene,
        float lightPosition
    );

    SceneModel(const SceneModel &other) = delete;
    SceneModel(SceneModel&& other) = delete;

    void subscribe(std::function<void(Vec3 color)> callback);

    void setColor(float r, float g, float b);
    Vec3 getColor() const;

    int getMaterialIndex() const;
    void setMaterialIndex(int materialIndex);

    void setLightPosition(float lightPosition);
    float getLightPosition() const;

    int getSpp() const;

private:
    const PathTracer *m_pathTracer;
    const Scene *m_scene;

    int m_materialIndex;

    int m_spp;
    float m_lightPosition;

    std::function<void(Vec3 color)> m_callback;
};

}
