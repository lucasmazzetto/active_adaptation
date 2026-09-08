# Mirror the environment that /isaac-sim/python.sh sets up, so plain `python`,
# `python -m pip`, and the `aa-*` console scripts work inside the container.
#
# setup_python_env.sh appends to PYTHONPATH and LD_LIBRARY_PATH without a default,
# which aborts under `set -u`. Seed both first so this file is safe to source from
# a strict shell (the entrypoint runs with `set -euo pipefail`).
: "${PYTHONPATH:=}"
: "${LD_LIBRARY_PATH:=}"
export PYTHONPATH LD_LIBRARY_PATH

export ISAAC_PATH=/isaac-sim
export CARB_APP_PATH=/isaac-sim/kit
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
export RESOURCE_NAME=IsaacSim
export LD_PRELOAD=/isaac-sim/kit/libcarb.so   # WAR for missing libcarb.so, as in python.sh
export PATH=/isaac-sim/kit/python/bin:${PATH}

# Isaac Sim ships warp as the Kit extension omni.warp.core (1.13.0 for Isaac Sim 6.0),
# and its own extensions are built against exactly that version. Put it ahead of
# site-packages so a pip-installed `warp-lang` of another version can never shadow it.
_aa_warp_core=$(ls -d /isaac-sim/extscache/omni.warp.core-*/ 2>/dev/null | head -1)
if [ -n "${_aa_warp_core}" ]; then
    export PYTHONPATH="${_aa_warp_core%/}:${PYTHONPATH}"
fi
unset _aa_warp_core
