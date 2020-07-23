#include "render_session.h"

#include <fstream>

#include "camera.h"
#include "parsers/obj_parser.h"
#include "scene_data.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace rays {

RenderSession::RenderSession()
{
    m_pathTracer = std::make_unique<PathTracer>();

    m_cudaGlobals = std::make_unique<CUDAGlobals>();

    std::string sceneFilename("../scenes/cornell-box/CornellBox-Original.obj");
    ObjParser objParser(sceneFilename);
    SceneData sceneData = SceneAdapter::createSceneData(objParser);
    std::cout << "triangle count: " << sceneData.triangles.size() << std::endl;
    std::cout << "sphere count: " << sceneData.spheres.size() << std::endl;
    std::cout << "material count: " << sceneData.materials.size() << std::endl;

    m_scene = std::make_unique<Scene>(sceneData);
    m_sceneModel = std::make_unique<SceneModel>(
        m_pathTracer.get(),
        m_scene.get(),
        defaultLightPosition
    );
    m_sceneModel->subscribe([this](Vec3 albedo, Vec3 emit) {
        m_scene->setColor(m_sceneModel->getMaterialIndex(), albedo);
        m_scene->setEmit(m_sceneModel->getMaterialIndex(), emit);

        m_pathTracer->reset();

        checkCudaErrors(cudaMemcpy(
            m_cudaGlobals->d_materials,
            m_scene->getMaterialsData(),
            m_scene->getMaterialsSize(),
            cudaMemcpyHostToDevice
        ));

        m_cudaGlobals->copySceneData(m_scene->getSceneData());

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    });
}

void RenderSession::init(
    GLuint pbo,
    int width,
    int height
) {
    m_width = width;
    m_height = height;

    const Camera camera(
        Vec3(0.f, 1.f, 6.8f),
        Vec3(0.f, 1.f, 0.f),
        19.5f / 180.f * M_PI,
        { width, height }
    );
    m_cudaGlobals->copyCamera(camera);

    m_cudaGlobals->mallocWorld(m_scene->getSceneData());

    m_scene->init();
    checkCudaErrors(cudaMemcpy(
        m_cudaGlobals->d_materials,
        m_scene->getMaterialsData(),
        m_scene->getMaterialsSize(),
        cudaMemcpyHostToDevice
    ));

    m_cudaGlobals->copySceneData(m_scene->getSceneData());

    checkCudaErrors(cudaGetLastError());

    m_pathTracer->init(pbo, width, height);
}

SceneModel& RenderSession::getSceneModel()
{
    return *m_sceneModel;
}


}

#undef checkCudaErrors
