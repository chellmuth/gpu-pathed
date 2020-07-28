#include "path_tracer.h"
#include "render_session.h"
#include "scene_model.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(path_tracer, m) {
    using namespace rays;

    m.doc() = "Path Tracer";

    py::class_<Vec3>(m, "Vec3")
        .def("x", &Vec3::x)
        .def("y", &Vec3::y)
        .def("z", &Vec3::z)
        .def("r", &Vec3::r)
        .def("g", &Vec3::g)
        .def("b", &Vec3::b);

    py::class_<RenderState>(m, "RenderState")
        .def_readonly("isRendering", &RenderState::isRendering)
        .def_readonly("pbo", &RenderState::pbo);

    py::class_<RenderSession>(m, "RenderSession")
        .def(py::init<int, int>())
        .def("init", &RenderSession::init, "Initialize CUDA resources")
        .def("renderAsync", &RenderSession::renderAsync, "Render and update buffer")
        .def("pollRender", &RenderSession::pollRender, "Poll the render state")
        .def("hitTest", &RenderSession::hitTest, "Run a material hit test")
        .def("getWidth", &RenderSession::getWidth)
        .def("getHeight", &RenderSession::getHeight)
        .def(
            "getSceneModel",
            &RenderSession::getSceneModel,
            py::return_value_policy::reference_internal
        );

    py::enum_<RendererType>(m, "RendererType")
        .value("CUDA", RendererType::CUDA)
        .value("Optix", RendererType::Optix)
        .export_values();

    py::class_<SceneModel>(m, "SceneModel")
        .def("getRendererType", &SceneModel::getRendererType)
        .def("setRendererType", &SceneModel::setRendererType)
        .def("setColor", &SceneModel::setColor)
        .def("getColor", &SceneModel::getColor)
        .def("setEmit", &SceneModel::setEmit)
        .def("getEmit", &SceneModel::getEmit)
        .def("getMaterialIndex", &SceneModel::getMaterialIndex)
        .def("setLightPosition", &SceneModel::setLightPosition)
        .def("getLightPosition", &SceneModel::getLightPosition)
        .def("getCameraOrigin", &SceneModel::getCameraOrigin)
        .def("setCameraOrigin", &SceneModel::setCameraOrigin)
        .def("getCameraTarget", &SceneModel::getCameraTarget)
        .def("setCameraTarget", &SceneModel::setCameraTarget)
        .def("getCameraUp", &SceneModel::getCameraUp)
        .def("setCameraUp", &SceneModel::setCameraUp)
        .def("getMaxDepth", &SceneModel::getMaxDepth)
        .def("setMaxDepth", &SceneModel::setMaxDepth)
        .def("zoomCamera", &SceneModel::zoomCamera)
        .def("getSpp", &SceneModel::getSpp);
}
