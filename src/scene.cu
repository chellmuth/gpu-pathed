#include "scene.h"

#include <iostream>

#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/microfacet.h"
#include "materials/mirror.h"

namespace rays {

namespace SceneParameters {

SceneData getSceneData(int index)
{

    if (index == 0) {
        SceneAdapter::ParseRequest request;

        auto defaultMaterial = std::make_unique<LambertianParams>(
            Vec3(1.f, 0.f, 0.f), Vec3(0.f)
        );
        request.materialParams.push_back(std::move(defaultMaterial));
        {
            std::string sceneFilename("../scenes/cornell-glossy/CornellBox-Glossy.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            request.defaultMaterialIDs.push_back(0);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/box.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            auto boxParams = std::make_unique<GlassParams>(1.4f);
            request.materialParams.push_back(std::move(boxParams));
            request.defaultMaterialIDs.push_back(1);
        }
        {
            std::string sceneFilename("../scenes/cornell-glossy/ball.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            auto ballParams = std::make_unique<MicrofacetParams>(0.3f);
            request.materialParams.push_back(std::move(ballParams));
            request.defaultMaterialIDs.push_back(2);
        }
        {
            auto sphereMaterial = std::make_unique<LambertianParams>(
                Vec3(1.f), Vec3(0.f)
            );
            request.materialParams.push_back(std::move(sphereMaterial));

            Sphere sphere(Vec3(0.f, 0.7, 0.f), 0.2f, request.materialParams.size() - 1);
            request.spheres.push_back(sphere);
        }

        request.environmentLightParams = EnvironmentLightParams(
            "../scenes/assets/20060807_wells6_hd.exr"
        );

        return SceneAdapter::createSceneData(request);
    } else if (index == 1) {
        std::string sceneFilename("../scenes/bunny/bunny.obj");
        ObjParser objParser(sceneFilename);
        return SceneAdapter::createSceneData(objParser);
    } else if (index == 2) {
        SceneAdapter::ParseRequest request;

        {
            std::string sceneFilename("../scenes/tests/plane.obj");
            ObjParser objParser(sceneFilename);
            request.objParsers.push_back(objParser);

            auto planeMaterial = std::make_unique<LambertianParams>(
                Vec3(1.f), Vec3(0.f)
            );
            request.materialParams.push_back(std::move(planeMaterial));
            request.defaultMaterialIDs.push_back(0);
        }

        request.environmentLightParams = EnvironmentLightParams(
            "../scenes/assets/1_pixel_test.exr"
        );

        return SceneAdapter::createSceneData(request);
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
    } else if (index == 2) {
        return Camera(
            Vec3(0.f, 1.f, 6.8f),
            Vec3(0.f, 1.f, 0.f),
            Vec3(0.f, 1.f, 0.f),
            19.5f / 180.f * M_PI,
            resolution
        );
    } else {
        std::cerr << "Invalid scene index: " << index << std::endl;
        exit(1);
    }
}

}

}
