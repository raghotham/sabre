"""
Helper introspection utility.

Extracts signatures and documentation from helper classes and functions.
"""
import inspect
import logging
import textwrap
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _format_type(type_hint: Any) -> str:
    """Format type hint for display."""
    if type_hint is None or type_hint == inspect.Parameter.empty:
        return "Any"
    if hasattr(type_hint, '__name__'):
        return type_hint.__name__
    else:
        # Handle complex types like List[Dict], Optional[str], etc.
        type_str = str(type_hint)
        # Clean up typing module prefixes
        type_str = type_str.replace('typing.', '')
        return type_str


def get_function_description_flat(func: Callable, name: str = None, class_name: str = None) -> str:
    """
    Get full function description with signature and docstring.

    Formats like:
        def func_name(param: type) -> return_type
            \"\"\"
            Docstring here
            \"\"\"

    Or for static methods:
        class ClassName:
            @staticmethod
            def method_name(param: type) -> return_type
                \"\"\"
                Docstring here
                \"\"\"

    Args:
        func: Function to inspect
        name: Override function name (optional)
        class_name: Class name for static methods (optional)

    Returns:
        Formatted function description
    """
    try:
        sig = inspect.signature(func)
        func_name = name or func.__name__

        # Build parameter list
        params = []
        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'cls'
            if param_name in ('self', 'cls'):
                continue

            # Build parameter string
            param_str = param_name

            # Add type annotation
            type_name = _format_type(param.annotation)
            param_str += f": {type_name}"

            # Add default value if present
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, str):
                    param_str += f' = "{param.default}"'
                elif param.default is None:
                    param_str += " = None"
                else:
                    param_str += f" = {param.default}"

            params.append(param_str)

        # Build return type
        return_type = _format_type(sig.return_annotation)

        # Get docstring
        doc = inspect.getdoc(func) or "No docstring"

        # Format based on whether it's a static method or standalone function
        if class_name:
            # Static method format
            return (
                f"class {class_name}:\n"
                f"    @staticmethod\n"
                f"    def {func_name}({', '.join(params)}) -> {return_type}\n"
                f'        """\n'
                f"{textwrap.indent(doc, ' ' * 8)}\n"
                f'        """\n'
            )
        else:
            # Standalone function format
            return (
                f"def {func_name}({', '.join(params)}) -> {return_type}\n"
                f'    """\n'
                f"{textwrap.indent(doc, ' ' * 4)}\n"
                f'    """\n'
            )

    except Exception as e:
        logger.error(f"Failed to introspect {name or func.__name__}: {e}")
        return f"def {name or func.__name__}(...)\n    \"\"\"\n    Function signature unavailable\n    \"\"\"\n"


def get_class_methods(cls: type, include_private: bool = False) -> list[str]:
    """
    Get all public methods from a class with full descriptions.

    Args:
        cls: Class to inspect
        include_private: Include private methods (default False)

    Returns:
        List of formatted function descriptions
    """
    descriptions = []
    class_name = cls.__name__

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip private methods unless requested
        if not include_private and name.startswith('_'):
            continue

        # Get full description with docstring
        desc = get_function_description_flat(method, name=name, class_name=class_name)
        descriptions.append(desc)

    return descriptions


def get_helper_signatures(namespace: dict) -> str:
    """
    Extract all helper descriptions from a namespace.

    Automatically discovers all classes and callable functions/objects
    in the namespace, without needing a hardcoded list.

    Args:
        namespace: Runtime namespace with helpers

    Returns:
        Formatted string with all helper descriptions
    """
    descriptions = []
    documented = set()

    # Skip these built-in names
    skip_names = {'print', 'pd', 'plt', 'datetime', '__builtins__'}

    # First pass: Document classes with static methods
    for name, obj in namespace.items():
        if name in skip_names or name.startswith('_'):
            continue

        if inspect.isclass(obj):
            # Document class methods
            class_descs = get_class_methods(obj)
            if class_descs:
                descriptions.extend(class_descs)
                documented.add(name)

    # Second pass: Document standalone functions and callable objects
    for name, obj in namespace.items():
        if name in skip_names or name.startswith('_') or name in documented:
            continue

        if callable(obj):
            try:
                # For callable class instances, use their __call__ method
                if hasattr(obj, '__call__') and not inspect.isfunction(obj) and not inspect.ismethod(obj):
                    desc = get_function_description_flat(obj.__call__, name=name)
                else:
                    desc = get_function_description_flat(obj, name=name)
                descriptions.append(desc)
                documented.add(name)
            except Exception as e:
                # If introspection fails, log but continue
                logger.debug(f"Skipped introspection for {name}: {e}")

    # Add available modules as comments
    modules = []
    if 'plt' in namespace:
        modules.append("# plt (matplotlib.pyplot) - for creating graphs")
    if 'pd' in namespace:
        modules.append("# pd (pandas) - for data manipulation")
    if 'datetime' in namespace:
        modules.append("# datetime (datetime module) - for date/time operations")

    result = '\n'.join(descriptions)
    if modules:
        result += '\n' + '\n'.join(modules)

    return result
