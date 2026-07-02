import bimanual
import numpy as np
np.set_printoptions(suppress=True)


if __name__ == "__main__":
	# joint 123456
	print(bimanual.forward_kinematics(np.array([0.6,0.0,0.0,0.0,0.0,0.0])))  #joint2pos
	# pos xyzrpy
	print(bimanual.inverse_kinematics(np.array([-0.01709269,0.05515691,0,0.00001469 ,0,0.6 ])))  #pos2joint
