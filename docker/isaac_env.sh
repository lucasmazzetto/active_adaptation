# Source Isaac Sim's Python environment (same variables as /isaac-sim/python.sh),
# so plain `python`, `pip`, and the `aa-*` console scripts work inside the container.
export ISAAC_PATH=/isaac-sim
export CARB_APP_PATH=/isaac-sim/kit
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
export PATH=/isaac-sim/kit/python/bin:${PATH}
