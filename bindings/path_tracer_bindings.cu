#include "path_tracer.h"
#include "render_session.h"
#include "scene_model.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(path_tracer, m) {
    using namespace rays;

    m.doc() = "Path Tracer";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def("x", &Vec3::x)
        .def("y", &Vec3::y)
        .def("z", &Vec3::z)
        .def("r", &Vec3::r)
        .def("g", &Vec3::g)
        .def("b", &Vec3::b);

    py::class_<RenderSession>(m, "RenderSession")
        .def(py::init<int, int>())
        .def("init", &RenderSession::init, "Initialize CUDA resources")
        .def("render", &RenderSession::render, "Render and update buffer")
        .def("hitTest", &RenderSession::hitTest, "Run a material hit test")
        .def("getWidth", &RenderSession::getWidth)
        .def("getHeight", &RenderSession::getHeight)
        .def(
            "getSceneModel",
            &RenderSession::getSceneModel,
            py::return_value_policy::reference_internal
        );

    py::class_<SceneModel>(m, "SceneModel")
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
        .def("getCameraUp", &SceneModel::getCameraUp)
        .def("getSpp", &SceneModel::getSpp);
}
