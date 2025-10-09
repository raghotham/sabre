import inspect
import os
from pathlib import Path
from typing import Any, Type, cast, Dict

import yaml


class Singleton (type):
    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Container(metaclass=Singleton):
    def __init__(
            self,
            config_file: str = os.path.expanduser('~/.config/llmvm/config.yaml'),
            throw: bool = True
        ):
        # First, load the default configuration
        default_config_path = Path(__file__).parent.parent / 'default_config.yaml'
        if default_config_path.exists():
            with open(default_config_path, 'r') as default_conf:
                self.configuration: Dict[str, Any] = yaml.load(default_conf, Loader=yaml.FullLoader) or {}
        else:
            self.configuration = {}

        self.config_file = config_file
        self.type_instance_cache: dict[Type, object] = {}

        if os.getenv('LLMVM_CONFIG'):
            self.config_file = cast(str, os.getenv('LLMVM_CONFIG'))

        # Load user config if it exists and merge with defaults
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as conf_file:
                user_config = yaml.load(conf_file, Loader=yaml.FullLoader) or {}
                # Deep merge user config over defaults
                self._deep_merge(self.configuration, user_config)
        elif not os.path.exists(self.config_file) and throw and not default_config_path.exists():
            raise ValueError('configuration_file {} is not found. Put config in ~/.config/llmvm or set LLMVM_CONFIG'.format(config_file))

    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        """Deep merge overlay dict into base dict"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def resolve(self, t: Type, **extra_args):
        args = {}
        for param in inspect.signature(t.__init__).parameters.values():
            if param == 'self':
                continue
            if extra_args and param.name in extra_args.keys():
                args[param.name] = extra_args[param.name]
            elif os.getenv(param.name.upper()):
                args[param.name] = os.getenv(param.name.upper())
            elif param.name in self.configuration and self.configuration[param.name]:
                args[param.name] = self.configuration[param.name]
        return t(**args)

    def get(self, key: str, default: Any = '') -> Any:
        if key not in self.configuration:
            return default

        value = self.configuration[key]
        if isinstance(value, str) and '~' in value and '/' in value:
            return os.path.expanduser(value)
        else:
            return value

    def has(self, key: str) -> bool:
        return key in self.configuration

    def config(self) -> dict:
        return self.configuration

    def resolve_cache(self, t: Type, **extra_args):
        if t in self.type_instance_cache:
            return self.type_instance_cache[t]
        else:
            self.type_instance_cache[t] = self.resolve(t, extra_args=extra_args)
            return self.type_instance_cache[t]

    @staticmethod
    def get_config_variable(name: str, alternate_name: str = '', default: Any = None) -> Any:
        def parse(value) -> Any:
            if isinstance(value, str) and (value == 'true' or value == 'True'):
                return True
            elif isinstance(value, str) and (value == 'false' or value == 'False'):
                return False
            elif isinstance(value, str) and value.lower() == 'none':
                return None
            elif isinstance(value, str) and str.isnumeric(value):
                return int(value)
            elif isinstance(value, str) and str.isdecimal(value):
                return float(value)
            else:
                return value

        if isinstance(default, str) and default.startswith('~'):
            default = os.path.expanduser(default)

        # environment variables take precendence
        if name in os.environ:
            return parse(os.environ.get(name, default))

        if alternate_name in os.environ:
            return parse(os.environ.get(alternate_name, default))

        # Try to get from singleton instance (which has merged defaults)
        try:
            container = Container(throw=False)
            # Check direct name first
            if container.has(name):
                return parse(container.get(name))
            # Check lowercase version without LLMVM_ prefix
            if container.has(name.replace('LLMVM_', '').lower()):
                return parse(container.get(name.replace('LLMVM_', '').lower()))
            # Check alternate name
            if alternate_name:
                if container.has(alternate_name):
                    return parse(container.get(alternate_name))
                if container.has(alternate_name.replace('LLMVM_', '').lower()):
                    return parse(container.get(alternate_name.replace('LLMVM_', '').lower()))
        except:
            pass

        # If all else fails, use provided default or empty string
        return parse(default) if default is not None else ''
