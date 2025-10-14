"""
Minimal Container stub for backwards compatibility with LLMVM prompt templates.

This module is only used by prompt_loader.py to provide a ContainerStub
for {{exec(...)}} template evaluations. It doesn't load any actual config files.
"""

import os


class Container:
    """
    Minimal container stub for prompt template compatibility.

    Note: This is not actively used for configuration in SABRE.
    All configuration is done via environment variables directly.
    """

    @staticmethod
    def get_config_variable(name: str, alternate_name: str = "", default: str = "") -> str:
        """
        Get configuration value from environment variables.

        Args:
            name: Primary environment variable name
            alternate_name: Alternate environment variable name
            default: Default value if not found

        Returns:
            Configuration value from environment or default
        """
        if isinstance(default, str) and default.startswith("~"):
            default = os.path.expanduser(default)

        # Check environment variables
        if name in os.environ:
            return os.environ.get(name, default)

        if alternate_name in os.environ:
            return os.environ.get(alternate_name, default)

        return default if default is not None else ""
