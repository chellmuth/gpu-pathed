#include "scene_data.h"

#include <assert.h>

#include "vec3.h"

namespace rays { namespace SceneAdapter {

SceneData createSceneData(ParseRequest &request)
{
    assert(request.objParsers.size() == request.defaultMaterials.size());

    SceneData sceneData;
    int materialOffset = 0;

    const int requestSize = request.objParsers.size();
    for (int i = 0; i < requestSize; i++) {
        ObjParser &objParser = request.objParsers[i];
        ObjResult result = objParser.parse();

        // Process materials
        if (result.mtls.empty()) {
            sceneData.materials.push_back(
                request.defaultMaterials[i]
            );
        }

        for (auto &mtl : result.mtls) {
            sceneData.materials.push_back(
                Material(
                    Vec3(mtl.r, mtl.g, mtl.b),
                    Vec3(mtl.emitR, mtl.emitG, mtl.emitB)
                )
            );
        }

        // Process geometry
        size_t faceCount = result.faces.size();
        for (size_t i = 0; i < faceCount; i++) {
            Face &face = result.faces[i];
            int materialIndex = result.mtlIndices[i];

            sceneData.triangles.push_back(
                Triangle(
                    Vec3(face.v0.x, face.v0.y, face.v0.z),
                    Vec3(face.v1.x, face.v1.y, face.v1.z),
                    Vec3(face.v2.x, face.v2.y, face.v2.z),
                    Vec3(face.n0.x, face.n0.y, face.n0.z),
                    Vec3(face.n1.x, face.n1.y, face.n1.z),
                    Vec3(face.n2.x, face.n2.y, face.n2.z),
                    materialOffset + materialIndex
                )
            );
        }

        materialOffset = sceneData.materials.size();
    }

    // Post-process lights
    size_t triangleCount = sceneData.triangles.size();
    for (size_t i = 0; i < triangleCount; i++) {
        size_t materialIndex = sceneData.triangles[i].materialIndex();
        if (sceneData.materials[materialIndex].getEmit().isZero()) { continue; }
        sceneData.lightIndices.push_back(i);
    }

    return sceneData;
}

SceneData createSceneData(ObjParser &objParser)
{
    ParseRequest request;

    request.objParsers.push_back(objParser);

    Material dummyMaterial(Vec3(0.f), Vec3(100.f, 0.f, 0.f));
    request.defaultMaterials.push_back(dummyMaterial);

    return createSceneData(request);
}

} }
