#include "scene_model.h"

namespace rays {

SceneModel::SceneModel(
    const PathTracer *pathTracer,
    const Scene *scene,
    float lightPosition
) : m_pathTracer(pathTracer),
    m_scene(scene),
    m_lightPosition(lightPosition),
    m_spp(0),
    m_materialIndex(-1)
{}

void SceneModel::subscribe(std::function<void(Vec3 albedo, Vec3 emit, Camera camera)> callback)
{
    m_callback = callback;
}

void SceneModel::setColor(float r, float g, float b)
{
    m_callback(Vec3(r, g, b), getEmit(), m_scene->getCamera());
}

Vec3 SceneModel::getColor() const
{
    if (m_materialIndex == -1) { return Vec3(0.f); }

    return m_scene->getMaterial(m_materialIndex).getAlbedo();
}

void SceneModel::setEmit(float r, float g, float b)
{
    m_callback(getColor(), Vec3(r, g, b), m_scene->getCamera());
}

Vec3 SceneModel::getEmit() const
{
    if (m_materialIndex == -1) { return Vec3(0.f); }

    return m_scene->getMaterial(m_materialIndex).getEmit();
}

int SceneModel::getMaterialIndex() const
{
    return m_materialIndex;
}

void SceneModel::setMaterialIndex(int materialIndex)
{
    m_materialIndex = materialIndex;
}

void SceneModel::setLightPosition(float lightPosition)
{
    m_lightPosition = lightPosition;
    m_callback(getColor(), getEmit(), m_scene->getCamera());
}

float SceneModel::getLightPosition() const
{
    return m_lightPosition;
}

int SceneModel::getSpp() const
{
    return m_pathTracer->getSpp();
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
    m_callback(getColor(), getEmit(), updated);
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
    m_callback(getColor(), getEmit(), updated);
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
    m_callback(getColor(), getEmit(), updated);
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
    m_callback(getColor(), getEmit(), updated);
}

}