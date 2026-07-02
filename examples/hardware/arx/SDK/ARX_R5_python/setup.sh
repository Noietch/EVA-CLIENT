#!/bin/bash
if [ -f /opt/ros/jazzy/setup.sh ]; then
    . /opt/ros/jazzy/setup.sh
fi

export LD_LIBRARY_PATH=$(pwd)/bimanual/api/arx_r5_src:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/bimanual/api:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
