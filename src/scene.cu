#include "scene.h"
namespace rays {

void Scene::init()
{
    update();
}

void Scene::update()
{
    m_materials = m_sceneData.materials;
}

}
