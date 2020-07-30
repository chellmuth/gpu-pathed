#include "scene_data.h"

#include "vec3.h"

namespace rays { namespace SceneAdapter {

SceneData createSceneData(ObjParser &objParser)
{
    ObjResult result = objParser.parse();

    SceneData sceneData;

    // Process materials
    if (result.mtls.empty()) {
        sceneData.materials.push_back(
            Material(Vec3(1.f), Vec3(0.1f))
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
                materialIndex
            )
        );
    }

    // Process lights
    for (size_t i = 0; i < faceCount; i++) {
        size_t materialIndex = sceneData.triangles[i].materialIndex();
        if (sceneData.materials[materialIndex].getEmit().isZero()) { continue; }

        sceneData.lightIndices.push_back(i);
    }

    return sceneData;
}

static Vec3 rotateY(Vec3 vector, float theta)
{
    return Vec3(
        cos(theta) * vector.x() - sin(theta) * vector.z(),
        vector.y(),
        sin(theta) * vector.x() + cos(theta) * vector.z()
    );
}

SceneData createSceneData(float lightPosition)
{
    SceneData sceneData;

    sceneData.spheres.push_back(
        Sphere(
            Vec3(0.f, 0.f, -1.f),
            0.8f,
            0
        )
    );
    sceneData.spheres.push_back(
        Sphere(
            Vec3(0.f, -100.8f, -1.f),
            100.f,
            2
        )
    );

    const float theta = lightPosition * M_PI;
    sceneData.triangles.push_back(
        Triangle(
            rotateY(Vec3(-0.5f, 0.6f, -2.f), theta),
            rotateY(Vec3(0.3f, 0.6f, -2.f), theta),
            rotateY(Vec3(-0.5f, 1.2f, -1.8f), theta),
            1
        )
    );
    sceneData.triangles.push_back(
        Triangle(
            rotateY(Vec3(0.3f, 0.6f, -2.f), theta),
            rotateY(Vec3(0.3f, 1.2f, -1.8f), theta),
            rotateY(Vec3(-0.5f, 1.2f, -1.8f), theta),
            1
        )
    );

    sceneData.materials.push_back(Material(Vec3(0.45098f, 0.823529f, 0.0862745f)));
    sceneData.materials.push_back(Material(Vec3(1.f, 0.2f, 1.f), Vec3(14.f, 14.f, 14.f)));
    sceneData.materials.push_back(Material(Vec3(1.f, 1.f, 1.f)));

    return sceneData;
}

} }
