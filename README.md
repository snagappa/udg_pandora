# POSSIBLE DEPENDENCIES
Python-numpy and python-scipy must be installed to run the slam node.
A blas wrapper used in the node links to the following:
libblas, liblapack, libgsl (and libgslcblas)

Most dependencies will be accounted for by install libatlas-dev and libgsl-dev. Additionally, you may require liblapack-dev and libblas-dev

The simulator depends on (recent versions of) gtk and glade libraries. The Python interface to gtk and glade is required as well.

## BLAS DEBUG:
Set lib.common.blas.DEBUG to False to disable basic checks on function arguments. This reduces function overhead, but can result in segfaults or *silent failures* if incorrect arguments are specified. Recommend that DEBUG be disabled only when all testing is complete.


## FOR SCI-KIT-LEARN

    sudo pip install scikit-learn


# TO RUN CHAIN SIMULATION SCENARIO

    roslaunch udg_pandora_sim valve_turning_udg.launch
    rosrun rviz rviz --> File --> Open Config --> udg_pandora/udg_pandora_misc/config/chain_setup.rviz
    rosservice call /cola2_control/keep_z "z: 4.4 altitude_mode: false"
    rosrun udg_pandora_sim sim_link_detector.py
    rosrun udg_pandora_chain chain_planner.py
    rosrun udg_pandora_chain chain_follow.py