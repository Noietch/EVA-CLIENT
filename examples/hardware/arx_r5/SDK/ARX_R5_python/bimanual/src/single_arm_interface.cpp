#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "arx_r5_src/interfaces/InterfacesPy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(arx_r5_python, m) {
    py::class_<arx::r5::InterfacesPy>(m, "InterfacesPy")
        .def(py::init<std::string,std::string,int>()) 
        .def("set_joint_positions", &arx::r5::InterfacesPy::set_joint_positions)
        .def("set_ee_pose", &arx::r5::InterfacesPy::set_ee_pose)
        .def("set_arm_status", &arx::r5::InterfacesPy::set_arm_status)
        .def("set_catch", &arx::r5::InterfacesPy::set_catch)
        .def("get_joint_positions", &arx::r5::InterfacesPy::get_joint_positions)
        .def("get_joint_velocities", &arx::r5::InterfacesPy::get_joint_velocities)
        .def("get_joint_currents", &arx::r5::InterfacesPy::get_joint_currents)
        .def("arx_x", &arx::r5::InterfacesPy::arx_x)
        .def("get_ee_pose", &arx::r5::InterfacesPy::get_ee_pose);
}
