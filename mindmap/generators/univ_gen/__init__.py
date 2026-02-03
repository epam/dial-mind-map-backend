"""
Mind Map Generator package initializer.

Sets up initial configuration, including loading environment variables.
"""

from .common.package_configurator import PackageConfigurator as _Configurator

# This configuration must execute before importing other modules that
# may depend on environment variables at import time.
_Configurator.configure()

from .mind_map_generator import MindMapGenerator

__all__ = ["MindMapGenerator"]
