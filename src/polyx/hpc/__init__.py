"""Environment tooling for the headless HPC training pipeline.

This package is Qt-free and importable on a compute node with no display. It
carries only the V100 environment contract: the CUDA/AMP probe, hash
verification of downloaded artifacts, and the manifest-driven weight fetcher.
The training pipeline itself lives elsewhere and is not imported here.
"""
