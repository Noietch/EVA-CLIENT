cmake_minimum_required(VERSION 3.10)
project(arx_r5_python VERSION 1.0)

find_package(pybind11 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/arx_r5_src/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/arx_hardware_interface/include)

pybind11_add_module(${PROJECT_NAME} src/single_arm_interface.cpp)
pybind11_add_module(kinematic_solver src/kinematic_solver.cpp)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Target architecture is ARM")
    set(DEPEND_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_r5_src/libarx_r5_src-arm64.so")
else ()
    set(DEPEND_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_r5_src/libarx_r5_src.so")
endif ()

target_include_directories(kinematic_solver PUBLIC lib/)
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPEND_LIB} pybind11::module)
target_link_libraries(kinematic_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/libkinematic_solver.so pybind11::module)

# 设置安装路径
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/api/" CACHE PATH "Installation Directory" FORCE)

install(DIRECTORY lib/ DESTINATION ./)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ./${PROJECT_NAME}/
        ARCHIVE DESTINATION ./${PROJECT_NAME}/)
install(TARGETS kinematic_solver
        LIBRARY DESTINATION ./${PROJECT_NAME}/
        ARCHIVE DESTINATION ./${PROJECT_NAME}/)
