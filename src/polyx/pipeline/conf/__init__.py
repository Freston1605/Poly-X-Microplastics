"""Hydra configuration tree for the headless pipeline.

Hydra resolves ``@main(config_path="conf")`` through the package that owns the
module, so this directory must be a package for the campaign and sweeper
multiruns to find their YAML files when they are invoked as modules.
"""
