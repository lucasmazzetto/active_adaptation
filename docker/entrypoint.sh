#!/usr/bin/env bash
# Runs on every container start.
set -euo pipefail
source /opt/isaac_env.sh
cd /workspace/active_adaptation

# The image already installs this repo editable at the same path the bind mount uses.
# This is only a fallback (e.g. if the image was built from a different context).
if ! python -m pip show active_adaptation >/dev/null 2>&1; then
    echo "[entrypoint] active_adaptation not installed in the image; installing now..."
    python -m pip install --no-deps -e /workspace/active_adaptation
fi

# Discover projects and enable only `facet`. The Hydra search-path plugin and the
# project importer both ignore disabled projects, so without this step neither the
# task YAML nor ImpedanceCommandManager is found.
aa-discover-projects >/dev/null
python - <<'PY'
import json, pathlib
p = pathlib.Path("/workspace/active_adaptation/.cache/projects.json")
d = json.loads(p.read_text())
for name, info in d.get("environment", {}).items():
    info["enabled"] = (name == "facet")
p.write_text(json.dumps(d, indent=2))
print("[entrypoint] projects:", {k: v["enabled"] for k, v in d["environment"].items()})
PY

exec "$@"
