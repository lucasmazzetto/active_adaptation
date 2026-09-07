# Prints pip constraint lines for the packages Isaac Sim bundles, so that installing this
# repository's dependencies cannot silently replace them.
#
# warp is deliberately NOT pinned here: Isaac Sim ships warp as a Kit extension whose version
# (1.8.2) is not published on PyPI, so an "==" pin would be unsatisfiable. The Dockerfile
# instead uninstalls the pip `warp-lang` after the dependency install and relies on the
# bundled extension, which isaac_env.sh puts on PYTHONPATH.
import importlib

for module_name, distribution in [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
]:
    try:
        version = importlib.import_module(module_name).__version__.split("+")[0]
        print(f"{distribution}=={version}")
    except Exception:
        pass
