from copy import deepcopy
from typing import Any, Dict, Tuple


LEFT_ARM_CONFIG: Dict[str, Any] = {
    "can_port": "can0",
    "type": 0,
}

RIGHT_ARM_CONFIG: Dict[str, Any] = {
    "can_port": "can1",
    "type": 0,
}


def get_dual_arm_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return deepcopy(LEFT_ARM_CONFIG), deepcopy(RIGHT_ARM_CONFIG)
