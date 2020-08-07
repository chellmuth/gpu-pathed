#include "scene.h"

#include <iostream>

#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/material_store.h"
#include "materials/mirror.h"

namespace rays {

namespace SceneParameters {

SceneData getSceneData(int index)
{

    if (index == 0) {
        SceneAdapter::ParseRequest request;
        MaterialStore store;

        Material testMaterial(Vec3(0.f), Vec3(0.f, 0.f, 0.f));
        const int testMaterialID = request.materialStore.addMaterial(testMaterial);

        {
            std::string sceneFilename("../scenes/cornell-glossy/CornellBox-Glossy.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            request.defaultMaterialIDs.push_back(testMaterialID);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/box.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Mirror boxMaterial;
            const int materialID = request.materialStore.addMaterial(boxMaterial);
            request.defaultMaterialIDs.push_back(materialID);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/ball.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Glass ballMaterial(1.4f);
            const int materialID = request.materialStore.addMaterial(ballMaterial);
            request.defaultMaterialIDs.push_back(materialID);
        }

        return SceneAdapter::createSceneData(request);
    } else if (index == 1) {
        std::string sceneFilename("../scenes/bunny/bunny.obj");
        ObjParser objParser(sceneFilename);
        return SceneAdapter::createSceneData(objParser);
    } else {
        std::cerr << "Invalid scene index: " << index << std::endl;
        exit(1);
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
        std::cerr << "Invalid scene index: " << index << std::endl;
        exit(1);
    }
}

}

void Scene::init()
{
    update();
}

void Scene::update()
{
    // m_lambertians = m_sceneData.lambertians;
    // m_mirrors = m_sceneData.mirrors;
}

}
