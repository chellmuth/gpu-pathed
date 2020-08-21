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
        {
            auto sphereMaterial = std::make_unique<LambertianParams>(
                Vec3(0.f, 1.f, 0.f), Vec3(0.f)
            );
            request.materialParams.push_back(std::move(sphereMaterial));

            Sphere sphere(Vec3(0.4f, 0.8, 0.f), 0.15f, request.materialParams.size() - 1);
            request.spheres.push_back(sphere);
        }

        // request.environmentLightParams = EnvironmentLightParams(
        //     "../scenes/assets/20060807_wells6_hd.exr"
        // );

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
    } else if (index == 3) {
        SceneAdapter::ParseRequest request;

        {
            const int materialOffset = request.materialParams.size();

            constexpr int paramCount = 5;
            std::unique_ptr<MaterialParams> paramsArray[paramCount] = {
                std::make_unique<LambertianParams>(Vec3(0.4f), Vec3(0.f)),
                std::make_unique<MicrofacetParams>(0.005f),
                std::make_unique<MicrofacetParams>(0.02f),
                std::make_unique<MicrofacetParams>(0.05f),
                std::make_unique<MicrofacetParams>(0.1f),
            };

            for (int i = 0; i < paramCount; i++) {
                request.materialParams.push_back(
                    std::move(paramsArray[i])
                );
            }

            std::vector<std::pair<std::string, int> > elements = {
                { "../scenes/mis/floor.ply", materialOffset + 0 },
                { "../scenes/mis/plate1.ply", materialOffset + 1 },
                { "../scenes/mis/plate2.ply", materialOffset + 2 },
                { "../scenes/mis/plate3.ply", materialOffset + 3 },
                { "../scenes/mis/plate4.ply", materialOffset + 4 },
            };

            for (const auto &element : elements) {
                const std::string sceneFilename = element.first;
                const int materialID = element.second;

                PLYParser plyParser(sceneFilename);
                request.plyParsers.push_back(plyParser);

                request.defaultMaterialIDs.push_back(materialID);
            }
        }
        {
            const int materialOffset = request.materialParams.size();
            std::vector<LambertianParams> materialParams = {
                LambertianParams(Vec3(0.f), Vec3(800.f)),
                LambertianParams(Vec3(0.f), Vec3(100.f)),
                LambertianParams(Vec3(0.f), Vec3(901.803f)),
                LambertianParams(Vec3(0.f), Vec3(11.1111f)),
                LambertianParams(Vec3(0.f), Vec3(1.234567f)),
            };

            for (const auto &params : materialParams) {
                request.materialParams.push_back(
                    std::make_unique<LambertianParams>(params)
                );
            }

            std::vector<Sphere> spheres = {
                Sphere(Vec3(10.f, 10.f, 4.f), 0.5f, materialOffset + 0),
                Sphere(Vec3(-1.25f, 0.f, 0.f), 0.1f, materialOffset + 1),
                Sphere(Vec3(-3.75f, 0.f, 0.f), 0.03333f, materialOffset + 2),
                Sphere(Vec3(1.25f, 0.f, 0.f), 0.3f, materialOffset + 3),
                Sphere(Vec3(3.75f, 0.f, 0.f), 0.9f, materialOffset + 4),
            };

            for (const auto &sphere : spheres) {
                request.spheres.push_back(sphere);
            }
        }

        request.environmentLightParams = EnvironmentLightParams(
            "../scenes/assets/20060807_wells6_hd.exr"
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
            resolution,
            false
        );
    } else if (index == 1) {
        return Camera(
            Vec3(0.f, 0.7f, 4.f),
            Vec3(0.f, 0.7f, 0.f),
            Vec3(0.f, 1.f, 0.f),
            28.f / 180.f * M_PI,
            resolution,
            false
        );
    } else if (index == 2) {
        return Camera(
            Vec3(0.f, 1.f, 6.8f),
            Vec3(0.f, 1.f, 0.f),
            Vec3(0.f, 1.f, 0.f),
            19.5f / 180.f * M_PI,
            resolution,
            false
        );
    } else if (index == 3) {
        return Camera(
            Vec3(0.f, 2.f, 15.f),
            Vec3(0.f, 1.69521f, 14.0476f),
            Vec3(0.f, 0.952421f, -0.304787f),
            28.0000262073138f / 180.f * M_PI,
            resolution,
            true
        );
    } else {
        std::cerr << "Invalid scene index: " << index << std::endl;
        exit(1);
    }
}

}

}
