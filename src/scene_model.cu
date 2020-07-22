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

void SceneModel::subscribe(std::function<void(Vec3 color)> callback)
{
    m_callback = callback;
}

void SceneModel::setColor(float r, float g, float b)
{
    m_callback(Vec3(r, g, b));
}

Vec3 SceneModel::getColor() const
{
    if (m_materialIndex == -1) { return Vec3(0.f); }

    return m_scene->getMaterial(m_materialIndex).getAlbedo();
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
    m_callback(getColor());
}

float SceneModel::getLightPosition() const
{
    return m_lightPosition;
}

int SceneModel::getSpp() const
{
    return m_pathTracer->getSpp();
}

}
