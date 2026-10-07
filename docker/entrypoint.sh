#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
    "$AA_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$WARP_CACHE_PATH" \
    "$OMNI_KIT_CACHE_ROOT" \
    "$OMNI_KIT_USER_CONFIG" \
    "$OMNI_KIT_USER_DATA" \
    "$HOME"

if [[ ! -f "$AA_CACHE_DIR/projects.json" ]]; then
    cat > "$AA_CACHE_DIR/projects.json" <<'JSON'
{
  "environment": {
    "facet": {
      "value": "projects.facet",
      "path": "/workspace/active_adaptation/projects/facet",
      "type": "environment",
      "enabled": true,
      "task_dir": "/workspace/active_adaptation/projects/facet/task",
      "config_path": "/workspace/active_adaptation/projects/facet"
    }
  },
  "learning": {}
}
JSON
fi

cd /workspace/active_adaptation
exec "$@"
