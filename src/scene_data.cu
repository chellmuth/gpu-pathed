#include "scene_data.h"

#include <assert.h>

#include "core/vec3.h"

namespace rays { namespace SceneAdapter {

SceneData createSceneData(ParseRequest &request)
{
    // assert(request.objParsers.size() == request.defaultMaterials.size());

    SceneData sceneData;

    {
        const int requestSize = request.objParsers.size();
        for (int i = 0; i < requestSize; i++) {
            ObjParser &objParser = request.objParsers[i];
            ObjResult result = objParser.parse();

            // Process materials
            const bool useDefaultMaterial = result.mtls.empty();

            int paramsOffset = request.materialParams.size();
            for (auto &mtl : result.mtls) {
                auto mtlParams = std::make_unique<LambertianParams>(
                    Vec3(mtl.r, mtl.g, mtl.b),
                    Vec3(mtl.emitR, mtl.emitG, mtl.emitB)
                );
                request.materialParams.push_back(std::move(mtlParams));
            }

            // Process geometry
            size_t faceCount = result.faces.size();
            for (size_t j = 0; j < faceCount; j++) {
                Face &face = result.faces[j];

                int materialID;
                if (useDefaultMaterial) {
                    materialID = request.defaultMaterialIDs[i];
                } else {
                    int mtlIndex = result.mtlIndices[j];
                    materialID = mtlIndex + paramsOffset;
                }

                sceneData.triangles.push_back(
                    Triangle(
                        Vec3(face.v0.x, face.v0.y, face.v0.z),
                        Vec3(face.v1.x, face.v1.y, face.v1.z),
                        Vec3(face.v2.x, face.v2.y, face.v2.z),
                        Vec3(face.n0.x, face.n0.y, face.n0.z),
                        Vec3(face.n1.x, face.n1.y, face.n1.z),
                        Vec3(face.n2.x, face.n2.y, face.n2.z),
                        materialID
                    )
                );
            }
        }
    }
    {
        const int requestSize = request.plyParsers.size();
        for (int i = 0; i < requestSize; i++) {
            PLYParser &plyParser = request.plyParsers[i];
            PLYResult result = plyParser.parse();

            // Process geometry
            size_t faceCount = result.faces.size();
            for (size_t j = 0; j < faceCount; j++) {
                Face &face = result.faces[j];

                int materialID = request.defaultMaterialIDs[i];

                sceneData.triangles.push_back(
                    Triangle(
                        Vec3(face.v0.x, face.v0.y, face.v0.z),
                        Vec3(face.v1.x, face.v1.y, face.v1.z),
                        Vec3(face.v2.x, face.v2.y, face.v2.z),
                        Vec3(face.n0.x, face.n0.y, face.n0.z),
                        Vec3(face.n1.x, face.n1.y, face.n1.z),
                        Vec3(face.n2.x, face.n2.y, face.n2.z),
                        materialID
                    )
                );
            }
        }
    }

    sceneData.spheres = request.spheres;
    sceneData.materialParams = std::move(request.materialParams);
    sceneData.environmentLightParams = request.environmentLightParams;

    // Post-process lights
    size_t triangleCount = sceneData.triangles.size();
    for (size_t i = 0; i < triangleCount; i++) {
        int materialID = sceneData.triangles[i].materialID();
        if (sceneData.isEmitter(materialID)) {
            sceneData.lightIndices.push_back(i);
        }
    }

    return sceneData;
}

SceneData createSceneData(ObjParser &objParser)
{
    ParseRequest request;

    request.objParsers.push_back(objParser);

    auto defaultMaterial = std::make_unique<LambertianParams>(
        Vec3(0.f), Vec3(100.f, 0.f, 0.f)
    );
    request.materialParams.push_back(std::move(defaultMaterial));
    request.defaultMaterialIDs.push_back(0);

    return createSceneData(request);
}

} }
