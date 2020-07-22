#include "render_session.h"

#include "camera.h"

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
    m_scene = std::make_unique<Scene>();
    m_sceneModel = std::make_unique<SceneModel>(
        m_pathTracer.get(),
        m_scene.get(),
        defaultLightPosition
    );
    m_sceneModel->subscribe([this](Vec3 color) {
        m_scene->setColor(m_sceneModel->getMaterialIndex(), color);
        m_pathTracer->reset();

        checkCudaErrors(cudaMemcpy(
            dev_materials,
            m_scene->getMaterialsData(),
            m_scene->getMaterialsSize(),
            cudaMemcpyHostToDevice
        ));

        createWorld<<<1, 1>>>(
            dev_primitives,
            dev_materials,
            m_cudaGlobals->d_world,
            m_sceneModel->getLightPosition(),
            true
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

    checkCudaErrors(cudaMalloc((void **)&dev_primitives, primitiveCount * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&dev_materials, materialCount * sizeof(Material)));
    m_cudaGlobals->mallocWorld();

    m_scene->init();
    checkCudaErrors(cudaMemcpy(
        dev_materials,
        m_scene->getMaterialsData(),
        m_scene->getMaterialsSize(),
        cudaMemcpyHostToDevice
    ));

    createWorld<<<1, 1>>>(
        dev_primitives,
        dev_materials,
        m_cudaGlobals->d_world,
        m_sceneModel->getLightPosition(),
        false
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
