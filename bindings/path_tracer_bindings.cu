#include "path_tracer.h"
#include "scene_model.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(path_tracer, m) {
    using namespace rays;

    m.doc() = "Path Tracer";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def("r", &Vec3::r)
        .def("g", &Vec3::g)
        .def("b", &Vec3::b);

    py::class_<PathTracer>(m, "PathTracer")
        .def(py::init<>())
        .def("init", &PathTracer::init, "Initialize CUDA resources")
        .def("render", &PathTracer::render, "Render and update buffer")
        .def(
            "getSceneModel",
            &PathTracer::getSceneModel,
            py::return_value_policy::reference_internal
        );

    py::class_<SceneModel>(m, "SceneModel")
        .def("setColor", &SceneModel::setColor)
        .def("getColor", &SceneModel::getColor)
        .def("setLightPosition", &SceneModel::setLightPosition)
        .def("getLightPosition", &SceneModel::getLightPosition)
        .def("getSpp", &SceneModel::getSpp);
}
