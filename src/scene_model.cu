#include "scene_model.h"

namespace rays {

SceneModel::SceneModel(
    const Scene *scene,
    float lightPosition,
    RendererType rendererType
) : m_scene(scene),
    m_lightPosition(lightPosition),
    m_rendererType(rendererType),
    m_spp(0),
    m_materialIndex({MaterialType::Lambertian, INT_MAX}),
    m_materialID(-1)
{}

void SceneModel::subscribe(std::function<void(const SceneModelAttributes &attributes)> callback)
{
    m_callback = callback;
}

RendererType SceneModel::getRendererType() const {
    return m_rendererType;
}

void SceneModel::setRendererType(RendererType rendererType)
{
    m_rendererType = rendererType;
    m_callback({
        getColor(),
        getEmit(),
        m_scene->getCamera(),
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

void SceneModel::setColor(float r, float g, float b)
{
    m_callback({
        Vec3(r, g, b),
        getEmit(),
        m_scene->getCamera(),
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

Vec3 SceneModel::getColor() const
{
    if (m_materialID == -1) { return Vec3(0.f); }
    return m_scene->getColor(m_materialID);
}

void SceneModel::setEmit(float r, float g, float b)
{
    m_callback({
        getColor(),
        Vec3(r, g, b),
        m_scene->getCamera(),
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

Vec3 SceneModel::getEmit() const
{
    if (m_materialID == -1) { return Vec3(0.f); }
    return m_scene->getEmit(m_materialID);
}

int SceneModel::getMaterialID() const
{
    return m_materialID;
}

void SceneModel::setMaterialID(int materialID)
{
    m_materialID = materialID;
}

MaterialIndex SceneModel::getMaterialIndex() const
{
    return m_materialIndex;
}

void SceneModel::setMaterialIndex(MaterialIndex materialIndex)
{
    m_materialIndex = materialIndex;
}

void SceneModel::setLightPosition(float lightPosition)
{
    m_lightPosition = lightPosition;
    m_callback({
        getColor(),
        getEmit(),
        m_scene->getCamera(),
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

float SceneModel::getLightPosition() const
{
    return m_lightPosition;
}

void SceneModel::updateSpp(int spp)
{
    m_spp = spp;
}

int SceneModel::getSpp() const
{
    return m_spp;
}

Vec3 SceneModel::getCameraOrigin() const
{
    return m_scene->getCamera().getOrigin();
}

void SceneModel::setCameraOrigin(float originX, float originY, float originZ)
{
    Vec3 origin(originX, originY, originZ);

    const Camera &current = m_scene->getCamera();
    Camera updated(
        origin,
        current.getTarget(),
        current.getUp(),
        current.getVerticalFOV(),
        current.getResolution()
    );
    m_callback({
        getColor(),
        getEmit(),
        updated,
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

Vec3 SceneModel::getCameraTarget() const
{
    return m_scene->getCamera().getTarget();
}

void SceneModel::setCameraTarget(float targetX, float targetY, float targetZ)
{
    Vec3 target(targetX, targetY, targetZ);

    const Camera &current = m_scene->getCamera();
    Camera updated(
        current.getOrigin(),
        target,
        current.getUp(),
        current.getVerticalFOV(),
        current.getResolution()
    );
    m_callback({
        getColor(),
        getEmit(),
        updated,
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

Vec3 SceneModel::getCameraUp() const
{
    return m_scene->getCamera().getUp();
}

void SceneModel::setCameraUp(float upX, float upY, float upZ)
{
    Vec3 up(upX, upY, upZ);

    const Camera &current = m_scene->getCamera();
    Camera updated(
        current.getOrigin(),
        current.getTarget(),
        up,
        current.getVerticalFOV(),
        current.getResolution()
    );
    m_callback({
        getColor(),
        getEmit(),
        updated,
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

void SceneModel::zoomCamera(float ticks)
{
    const Camera &current = m_scene->getCamera();
    const Vec3 tickDirection = normalized(current.getTarget() - current.getOrigin());

    Camera updated(
        current.getOrigin() + tickDirection * ticks,
        current.getTarget(),
        current.getUp(),
        current.getVerticalFOV(),
        current.getResolution()
    );
    m_callback({
        getColor(),
        getEmit(),
        updated,
        m_rendererType,
        getMaxDepth(),
        getNextEventEstimation()
    });
}

int SceneModel::getMaxDepth() const
{
    return m_scene->getMaxDepth();
}

void SceneModel::setMaxDepth(int maxDepth)
{
    m_callback({
        getColor(),
        getEmit(),
        m_scene->getCamera(),
        m_rendererType,
        maxDepth,
        getNextEventEstimation()
    });
}

bool SceneModel::getNextEventEstimation() const
{
    return m_scene->getNextEventEstimation();
}

void SceneModel::setNextEventEstimation(bool nextEventEstimation)
{
    m_callback({
        getColor(),
        getEmit(),
        m_scene->getCamera(),
        m_rendererType,
        getMaxDepth(),
        nextEventEstimation
    });
}

}
