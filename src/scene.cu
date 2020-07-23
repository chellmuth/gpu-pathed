#include "scene.h"
namespace rays {

namespace SceneParameters {

SceneData getSceneData(int index)
{
    if (index == 0) {
        std::string sceneFilename("../scenes/cornell-box/CornellBox-Original.obj");
        ObjParser objParser(sceneFilename);
        return SceneAdapter::createSceneData(objParser);
    } else {
        return SceneAdapter::createSceneData(defaultLightPosition);
    }
}

Camera getCamera(int index, Resolution resolution)
{
    if (index == 0) {
        return Camera(
            Vec3(0.f, 1.f, 6.8f),
            Vec3(0.f, 1.f, 0.f),
            19.5f / 180.f * M_PI,
            resolution
        );
    } else {
        return Camera(
            Vec3(0.f, 0.3f, 5.f),
            Vec3(0.f),
            30.f / 180.f * M_PI,
            resolution
        );
    }
}

}

void Scene::init()
{
    update();
}

void Scene::update()
{
    m_materials = m_sceneData.materials;
}

}
