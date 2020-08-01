#include "scene.h"

#include "material.h"

namespace rays {

namespace SceneParameters {

SceneData getSceneData(int index)
{
    Material dummyMaterial(Vec3(0.f), Vec3(100.f, 0.f, 0.f));

    if (index == 0) {
        SceneAdapter::ParseRequest request;

        {
            std::string sceneFilename("../scenes/cornell-glossy/CornellBox-Glossy.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);
            request.defaultMaterials.push_back(dummyMaterial);
        }
        {
            Material defaultMaterial(
                Vec3(1.f, 0.f, 0.f),
                Vec3(0.f)
            );
            std::string sceneFilename("../scenes/cornell-glossy/box.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Material boxMaterial(Vec3(1.f, 0.f, 0.f), Vec3(0.f));
            request.defaultMaterials.push_back(boxMaterial);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/ball.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Material sphereMaterial(Vec3(1.f, 1.f, 0.f), Vec3(0.f));
            request.defaultMaterials.push_back(sphereMaterial);
        }

        return SceneAdapter::createSceneData(request);
    } else if (index == 1) {
        std::string sceneFilename("../scenes/bunny/bunny.obj");
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
            Vec3(0.f, 1.f, 0.f),
            19.5f / 180.f * M_PI,
            resolution
        );
    } else if (index == 1) {
        return Camera(
            Vec3(0.f, 0.7f, 4.f),
            Vec3(0.f, 0.7f, 0.f),
            Vec3(0.f, 1.f, 0.f),
            28.f / 180.f * M_PI,
            resolution
        );
    } else {
        return Camera(
            Vec3(0.f, 0.3f, 5.f),
            Vec3(0.f),
            Vec3(0.f, 1.f, 0.f),
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
