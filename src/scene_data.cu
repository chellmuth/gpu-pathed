#include "scene_data.h"

#include <assert.h>

#include "vec3.h"

namespace rays { namespace SceneAdapter {

SceneData createSceneData(ParseRequest &request)
{
    // assert(request.objParsers.size() == request.defaultMaterials.size());

    SceneData sceneData;
    std::vector<int> materialIDs;

    const int requestSize = request.objParsers.size();
    for (int i = 0; i < requestSize; i++) {
        ObjParser &objParser = request.objParsers[i];
        ObjResult result = objParser.parse();

        // Process materials
        const bool useDefaultMaterial = result.mtls.empty();

        for (auto &mtl : result.mtls) {
            Lambertian objMaterial(
                Vec3(mtl.r, mtl.g, mtl.b),
                Vec3(mtl.emitR, mtl.emitG, mtl.emitB)
            );

            const int materialID = request.materialStore.addMaterial(objMaterial);
            materialIDs.push_back(materialID);
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
                materialID = materialIDs[mtlIndex];
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

    sceneData.materialStore = request.materialStore;

    // Post-process lights
    size_t triangleCount = sceneData.triangles.size();
    for (size_t i = 0; i < triangleCount; i++) {
        int materialID = sceneData.triangles[i].materialID();
        MaterialIndex materialIndex = sceneData.materialStore.indexAt(materialID);
        if (sceneData.isEmitter(materialIndex)) {
            sceneData.lightIndices.push_back(i);
        }
    }

    return sceneData;
}

SceneData createSceneData(ObjParser &objParser)
{
    ParseRequest request;
    MaterialStore store;

    request.objParsers.push_back(objParser);

    Lambertian defaultMaterial(Vec3(0.f), Vec3(100.f, 0.f, 0.f));
    const int defaultMaterialID = request.materialStore.addMaterial(defaultMaterial);

    return createSceneData(request);
}

} }
