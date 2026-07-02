#include <string>
#include <memory>

namespace arx {
class KinematicSolver {
 public:
  KinematicSolver();
  ~KinematicSolver();
  void computeForwardKinematics(double joint_angles[6], double end_effector_pose[6]);
  void computeInverseKinematics(double target_pose[6], double joint_angles[6]);
 private:
  class impl;
  std::unique_ptr<impl> pimpl;
};
}