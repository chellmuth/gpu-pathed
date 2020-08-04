#include "scene.h"

#include <iostream>

#include "materials/mirror.h"
#include "materials/material.h"
#include "materials/material_store.h"

namespace rays {

namespace SceneParameters {

SceneData getSceneData(int index)
{
    Material mirrorMaterial(Vec3(0.f), Vec3(0.f, 0.f, 0.f));

    if (index == 0) {
        SceneAdapter::ParseRequest request;
        MaterialStore store;

        {
            std::string sceneFilename("../scenes/cornell-glossy/CornellBox-Glossy.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);
            request.defaultMaterials.push_back(mirrorMaterial);

            const MaterialIndex index = request.materialTable.addMaterial(mirrorMaterial);
            request.defaultMaterialIndices.push_back(index);

            const int materialID = request.materialStore.addMaterial(mirrorMaterial);
            request.defaultMaterialIDs.push_back(materialID);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/box.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Material boxMaterial(Vec3(1.f, 0.f, 1.f), Vec3(0.f));
            request.defaultMaterials.push_back(boxMaterial);

            const MaterialIndex index = request.materialTable.addMaterial(boxMaterial);
            request.defaultMaterialIndices.push_back(index);

            const int materialID = request.materialStore.addMaterial(boxMaterial);
            request.defaultMaterialIDs.push_back(materialID);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/ball.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Mirror ballMaterial;
            Material fakeDefault(Vec3(1.f), Vec3(10.f));
            request.defaultMaterials.push_back(fakeDefault);

            const MaterialIndex index = request.materialTable.addMaterial(ballMaterial);
            request.defaultMaterialIndices.push_back(index);

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
    m_materials = m_sceneData.materials;
    m_lambertians = m_sceneData.lambertians;
    m_mirrors = m_sceneData.mirrors;
}

}
