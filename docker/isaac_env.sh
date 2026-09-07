# Mirror the environment that /isaac-sim/python.sh sets up, so plain `python`,
# `python -m pip`, and the `aa-*` console scripts work inside the container.
export ISAAC_PATH=/isaac-sim
export CARB_APP_PATH=/isaac-sim/kit
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh   # sets PYTHONPATH and LD_LIBRARY_PATH
export RESOURCE_NAME=IsaacSim
export LD_PRELOAD=/isaac-sim/kit/libcarb.so   # WAR for missing libcarb.so, as in python.sh
export PATH=/isaac-sim/kit/python/bin:${PATH}
