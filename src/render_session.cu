#include "render_session.h"

#include <fstream>

#include "camera.h"
#include "parsers/obj_parser.h"

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

    std::ifstream sceneFile("../scenes/cornell-box/CornellBox-Original.obj");
    ObjParser objParser(sceneFile);
    m_scene = std::make_unique<Scene>();
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

        copyGeometry(
            m_cudaGlobals->d_triangles,
            m_cudaGlobals->d_spheres,
            m_cudaGlobals->d_materials,
            m_cudaGlobals->d_world,
            m_sceneModel->getLightPosition()
        );

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
        Vec3(0.f, 0.3f, 5.f),
        30.f / 180.f * M_PI,
        { width, height }
    );
    m_cudaGlobals->copyCamera(camera);

    m_cudaGlobals->mallocWorld();

    m_scene->init();
    checkCudaErrors(cudaMemcpy(
        m_cudaGlobals->d_materials,
        m_scene->getMaterialsData(),
        m_scene->getMaterialsSize(),
        cudaMemcpyHostToDevice
    ));

    copyGeometry(
        m_cudaGlobals->d_triangles,
        m_cudaGlobals->d_spheres,
        m_cudaGlobals->d_materials,
        m_cudaGlobals->d_world,
        m_sceneModel->getLightPosition()
    );

    checkCudaErrors(cudaGetLastError());

    m_pathTracer->init(pbo, width, height);
}

SceneModel& RenderSession::getSceneModel()
{
    return *m_sceneModel;
}


}

#undef checkCudaErrors
