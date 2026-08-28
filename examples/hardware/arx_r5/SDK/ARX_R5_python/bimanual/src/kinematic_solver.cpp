#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "kinematic_solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(kinematic_solver, m) {
    py::class_<arx::KinematicSolver>(m, "KinematicSolver")
        .def(py::init<>())
        .def("forward_kinematics", [](arx::KinematicSolver& self,py::array_t<double> in_arr){
        auto in_buf = in_arr.request();
        double* input = static_cast<double*>(in_buf.ptr);

        py::array_t<double> out_arr(6);
        auto out_buf = out_arr.request();
        double* output = static_cast<double*>(out_buf.ptr);

        self.computeForwardKinematics(input,output);
        return out_arr;
      })
        .def("inverse_kinematics",[](arx::KinematicSolver& self,py::array_t<double> in_arr){
        auto in_buf = in_arr.request();
        double* input = static_cast<double*>(in_buf.ptr);

        py::array_t<double> out_arr(6);
        auto out_buf = out_arr.request();
        double* output = static_cast<double*>(out_buf.ptr);

        self.computeInverseKinematics(input,output);
        return out_arr;
      });
}
