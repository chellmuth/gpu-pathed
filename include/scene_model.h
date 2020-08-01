#pragma once

#include <functional>
#include <iostream>

#include "material.h"
#include "path_tracer.h"
#include "scene.h"
#include "vec3.h"

namespace rays {

class OptixTracer;

enum class RendererType {
    CUDA,
    Optix,
    Normals
};

struct SceneModelAttributes {
    Vec3 albedo;
    Vec3 emitted;
    Camera camera;
    RendererType rendererType;
    int maxDepth;
    bool nextEventEstimation;
};

class SceneModel {
public:
    SceneModel(
        const Scene *scene,
        float lightPosition,
        RendererType rendererType
    );

    SceneModel(const SceneModel &other) = delete;
    SceneModel(SceneModel&& other) = delete;

    void subscribe(std::function<void(const SceneModelAttributes &attributes)> callback);

    RendererType getRendererType() const;
    void setRendererType(RendererType rendererType);

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

    void updateSpp(int spp);
    int getSpp() const;

    int getMaxDepth() const;
    void setMaxDepth(int maxDepth);

    bool getNextEventEstimation() const;
    void setNextEventEstimation(bool nextEventEstimation);

    void zoomCamera(float ticks);

private:
    const Scene *m_scene;

    RendererType m_rendererType;
    int m_materialIndex;

    int m_spp;
    float m_lightPosition;

    std::function<void(const SceneModelAttributes &attributes)> m_callback;
};

}
