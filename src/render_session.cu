#include "render_session.h"

#include <fstream>

#include "camera.h"
#include "macro_helper.h"
#include "renderers/optix.h"
#include "parsers/obj_parser.h"
#include "scene_data.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

void PBOManager::init(GLuint pbo1, GLuint pbo2)
{
    m_computeOnPrimary = false;

    m_pbo1 = pbo1;
    m_pbo2 = pbo2;

    checkCudaErrors(
        cudaGraphicsGLRegisterBuffer(
            &m_resource1,
            m_pbo1,
            cudaGraphicsRegisterFlagsWriteDiscard
        )
    );
    checkCudaErrors(
        cudaGraphicsGLRegisterBuffer(
            &m_resource2,
            m_pbo2,
            cudaGraphicsRegisterFlagsWriteDiscard
        )
    );
}

cudaGraphicsResource *PBOManager::getRenderResource()
{
    if (m_computeOnPrimary) { return m_resource1; }
    else { return m_resource2; }
}

GLuint PBOManager::getDisplayPBO()
{
    if (m_computeOnPrimary) { return m_pbo2; }
    else { return m_pbo1; }
}

void PBOManager::swapPBOs()
{
    m_computeOnPrimary = !m_computeOnPrimary;
}

RenderSession::RenderSession(int width, int height)
    : m_width(width),
      m_height(height),
      m_rendererType(RendererType::Optix)
{
    if (m_rendererType == RendererType::CUDA) {
        m_pathTracer = std::make_unique<PathTracer>();
    } else if (m_rendererType == RendererType::Optix) {
        m_pathTracer = std::make_unique<OptixTracer>();
    }
    m_cudaGlobals = std::make_unique<CUDAGlobals>();

    constexpr int sceneIndex = 0;
    SceneData sceneData = SceneParameters::getSceneData(sceneIndex);
    Camera camera = SceneParameters::getCamera(sceneIndex, { width, height });

    m_scene = std::make_unique<Scene>(
        camera,
        sceneData
    );
    m_sceneModel = std::make_unique<SceneModel>(
        m_scene.get(),
        defaultLightPosition,
        m_rendererType
    );
    m_sceneModel->subscribe([this](const SceneModelAttributes &attributes) {
        if (attributes.rendererType != m_rendererType) {
            m_rendererType = attributes.rendererType;
            m_resetRenderer = true;
            return;
        }

        m_scene->setMaxDepth(attributes.maxDepth);
        m_scene->setColor(m_sceneModel->getMaterialIndex(), attributes.albedo);
        m_scene->setEmit(m_sceneModel->getMaterialIndex(), attributes.emitted);

        m_scene->setCamera(attributes.camera);
        m_cudaGlobals->copyCamera(m_scene->getCamera());

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

RenderState RenderSession::init(GLuint pbo1, GLuint pbo2)
{
    m_pboManager.init(pbo1, pbo2);

    m_cudaGlobals->mallocCamera();
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

    m_pathTracer->init(m_width, m_height, *m_scene);

    return RenderState{false, pbo1};
}

SceneModel& RenderSession::getSceneModel()
{
    return *m_sceneModel;
}

}
