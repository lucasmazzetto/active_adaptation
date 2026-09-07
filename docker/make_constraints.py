# Prints pip constraint lines for the packages Isaac Sim bundles, so they are never upgraded.
import importlib

for name, dist in [("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"), ("warp", "warp-lang")]:
    try:
        version = importlib.import_module(name).__version__.split("+")[0]
        print(f"{dist}=={version}")
    except Exception:
        pass
