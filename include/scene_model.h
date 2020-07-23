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

    void subscribe(std::function<void(Vec3 albedo, Vec3 emit, Camera camera)> callback);

    void setColor(float r, float g, float b);
    Vec3 getColor() const;

    void setEmit(float r, float g, float b);
    Vec3 getEmit() const;

    int getMaterialIndex() const;
    void setMaterialIndex(int materialIndex);

    void setLightPosition(float lightPosition);
    float getLightPosition() const;

    Vec3 getCameraOrigin() const;
    void setCameraOrigin(float originX, float originY, float originZ);
    Vec3 getCameraTarget() const;
    void setCameraTarget(float targetX, float targetY, float targetZ);
    Vec3 getCameraUp() const;
    void setCameraUp(float upX, float upY, float upZ);

    int getSpp() const;

private:
    const PathTracer *m_pathTracer;
    const Scene *m_scene;

    int m_materialIndex;

    int m_spp;
    float m_lightPosition;

    std::function<void(Vec3 albedo, Vec3 emit, Camera camera)> m_callback;
};

}
