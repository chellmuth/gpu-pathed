#include "scene_model.h"

namespace rays {

SceneModel::SceneModel(
    const PathTracer *pathTracer,
    const Scene *scene,
    const Vec3 &color,
    float lightPosition
) : m_pathTracer(pathTracer),
    m_scene(scene),
    m_r(color.r()),
    m_g(color.g()),
    m_b(color.b()),
    m_lightPosition(lightPosition),
    m_spp(0),
    m_materialIndex(-1)
{}

void SceneModel::subscribe(std::function<void()> callback)
{
    m_callback = callback;
}

void SceneModel::setColor(float r, float g, float b)
{
    m_r = r;
    m_g = g;
    m_b = b;

    m_callback();
}

Vec3 SceneModel::getColor() const
{
    return Vec3(m_r, m_g, m_b);
}

int SceneModel::getMaterialIndex() const
{
    return m_materialIndex;
}

void SceneModel::setMaterialIndex(int materialIndex, const Vec3 &albedo)
{
    m_materialIndex = materialIndex;

    m_r = albedo.r();
    m_g = albedo.g();
    m_b = albedo.b();
}

void SceneModel::setLightPosition(float lightPosition)
{
    m_lightPosition = lightPosition;
    m_callback();
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
