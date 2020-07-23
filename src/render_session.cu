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

RenderSession::RenderSession(int width, int height)
    : m_width(width),
      m_height(height)
{
    m_pathTracer = std::make_unique<PathTracer>();

    m_cudaGlobals = std::make_unique<CUDAGlobals>();

    constexpr int sceneIndex = 0;
    SceneData sceneData = SceneParameters::getSceneData(sceneIndex);
    Camera camera = SceneParameters::getCamera(sceneIndex, { width, height });

    m_scene = std::make_unique<Scene>(
        camera,
        sceneData
    );
    m_sceneModel = std::make_unique<SceneModel>(
        m_pathTracer.get(),
        m_scene.get(),
        defaultLightPosition
    );
    m_sceneModel->subscribe([this](Vec3 albedo, Vec3 emit, Camera camera) {
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

void RenderSession::init(GLuint pbo)
{
    m_cudaGlobals->copyCamera(m_scene->getCamera());

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

    m_pathTracer->init(pbo, m_width, m_height);
}

SceneModel& RenderSession::getSceneModel()
{
    return *m_sceneModel;
}

}

#undef checkCudaErrors
