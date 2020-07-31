#pragma once

#include "cuda_globals.h"
#include "hit_test.h"
#include "material.h"
#include "path_tracer.h"
#include "render_record.h"
#include "renderers/optix_tracer.h"
#include "renderers/renderer.h"
#include "scene.h"
#include "scene_model.h"
#include "sphere.h"
#include "spp_optimizer.h"
#include "triangle.h"

namespace rays {

class PBOManager {
public:
    void init(GLuint pbo1, GLuint pbo2);
    cudaGraphicsResource *getRenderResource();
    GLuint getDisplayPBO();
    void swapPBOs();

private:
    bool m_computeOnPrimary;

    GLuint m_pbo1, m_pbo2;
    cudaGraphicsResource *m_resource1, *m_resource2;
};

struct RenderState {
    bool isRendering;
    GLuint pbo;
};

class RenderSession {
public:
    RenderSession(int width, int height);

    RenderSession(const RenderSession &other) = delete;
    RenderSession(RenderSession&& other) = delete;

    SceneModel& getSceneModel();

    RenderState init(GLuint pbo1, GLuint pbo2);

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

    void hitTest(int x, int y) {
        rays::hitTest(*m_sceneModel, *m_cudaGlobals, x, y, m_width, m_height);
    }

    RenderState renderAsync() {
        cudaGraphicsResource *pboResource = m_pboManager.getRenderResource();
        const int spp = m_sppOptimizer.estimateSpp();
        m_currentRecord = m_pathTracer->renderAsync(
            spp,
            pboResource,
            *m_scene,
            *m_cudaGlobals
        );
        return RenderState{true, m_pboManager.getDisplayPBO()};
    }

    RenderState pollRender() {
        cudaGraphicsResource *pboResource = m_pboManager.getRenderResource();
        bool finished = m_pathTracer->pollRender(pboResource, m_currentRecord);
        if (finished) {
            float milliseconds = 0;
            cudaEventElapsedTime(
                &milliseconds,
                m_currentRecord.beginEvent,
                m_currentRecord.endEvent
            );

            m_sppOptimizer.track(m_currentRecord.spp, milliseconds);

            m_pboManager.swapPBOs();
            m_sceneModel->updateSpp(m_pathTracer->getSpp());

            if (m_resetRenderer) {
                if (m_rendererType == RendererType::CUDA) {
                    m_pathTracer.reset(new PathTracer());
                } else if (m_rendererType == RendererType::Optix) {
                    m_pathTracer.reset(new OptixTracer());
                }
                m_pathTracer->init(m_width, m_height, *m_scene);
                m_resetRenderer = false;
            }
        }

        return RenderState{!finished, m_pboManager.getDisplayPBO()};
    }

private:
    RenderRecord m_currentRecord;
    SppOptimizer m_sppOptimizer;

    std::unique_ptr<Renderer> m_pathTracer;
    PBOManager m_pboManager;
    int m_width, m_height;

    bool m_resetRenderer;
    RendererType m_rendererType;

    std::unique_ptr<CUDAGlobals> m_cudaGlobals;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<SceneModel> m_sceneModel;
};

}
