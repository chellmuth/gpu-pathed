#include "scene.h"

#include <iostream>

#include "materials/dummy.h"
#include "materials/material.h"

namespace rays {

namespace SceneParameters {

SceneData getSceneData(int index)
{
    Material dummyMaterial(Vec3(0.f), Vec3(0.f, 0.f, 0.f));

    if (index == 0) {
        SceneAdapter::ParseRequest request;

        {
            std::string sceneFilename("../scenes/cornell-glossy/CornellBox-Glossy.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);
            request.defaultMaterials.push_back(dummyMaterial);

            const MaterialIndex index = request.materialTable.addMaterial(dummyMaterial);
            request.defaultMaterialIndices.push_back(index);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/box.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Material boxMaterial(Vec3(1.f, 0.f, 1.f), Vec3(0.f));
            request.defaultMaterials.push_back(boxMaterial);

            const MaterialIndex index = request.materialTable.addMaterial(boxMaterial);
            request.defaultMaterialIndices.push_back(index);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/ball.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            Dummy ballMaterial;
            Material fakeDefault(Vec3(1.f), Vec3(10.f));
            request.defaultMaterials.push_back(fakeDefault);

            const MaterialIndex index = request.materialTable.addMaterial(ballMaterial);
            request.defaultMaterialIndices.push_back(index);
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
    m_dummies = m_sceneData.dummies;
}

}
