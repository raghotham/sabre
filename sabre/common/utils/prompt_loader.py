"""
Prompt loader for sabre.

Loads and populates prompt templates with variable substitutions.
Supports:
- {{key}} replacements from template dict
- {{exec(...)}} dynamic evaluations (dates, config values, etc.)
"""

import os
import datetime
from typing import Dict, Any


class PromptLoader:
    """Loads and populates prompt templates."""

    @staticmethod
    def load(
        prompt_name: str, template: Dict[str, Any] | None = None, module_path: str = "sabre.server.prompts"
    ) -> Dict[str, str]:
        """
        Load and populate a prompt template.

        Args:
            prompt_name: Name of prompt file (e.g., 'python_continuation_execution.prompt')
            template: Dict of variables to substitute (optional)
            module_path: Module path where prompts are stored

        Returns:
            Dict with 'system_message' and 'user_message' keys

        Example:
            prompt = PromptLoader.load(
                'python_continuation_execution.prompt',
                template={
                    'functions': 'Bash.execute(...)',
                    'context_window_tokens': '128000',
                }
            )
        """
        template = template or {}

        # Load prompt file
        prompt_file = PromptLoader._find_prompt_file(prompt_name, module_path)
        with open(prompt_file, "r") as f:
            prompt_text = f.read()

        # Parse prompt sections
        if "[system_message]" not in prompt_text:
            raise ValueError("Prompt file must contain [system_message]")
        if "[user_message]" not in prompt_text:
            raise ValueError("Prompt file must contain [user_message]")

        system_start = prompt_text.find("[system_message]") + len("[system_message]")
        user_start = prompt_text.find("[user_message]")

        system_message = prompt_text[system_start:user_start].strip()
        user_message = prompt_text[user_start + len("[user_message]") :].strip()

        # Apply template substitutions
        result = {
            "system_message": PromptLoader._substitute(system_message, template),
            "user_message": PromptLoader._substitute(user_message, template),
        }

        return result

    @staticmethod
    def _find_prompt_file(prompt_name: str, module_path: str) -> str:
        """
        Find prompt file path.

        Args:
            prompt_name: Prompt filename
            module_path: Module path (e.g., 'sabre.server.prompts')

        Returns:
            Absolute path to prompt file
        """
        # Convert module path to file path
        # e.g., 'sabre.server.prompts' -> 'sabre/server/prompts'
        parts = module_path.split(".")

        # Get the base directory (where sabre package is)
        current_file = os.path.abspath(__file__)
        # Go up from: sabre/common/utils/prompt_loader.py
        sabre_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

        # Build path to prompts directory
        prompt_dir = sabre_root
        for part in parts:
            if part == "sabre":
                continue  # Already at sabre_root
            prompt_dir = os.path.join(prompt_dir, part)

        prompt_file = os.path.join(prompt_dir, prompt_name)

        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        return prompt_file

    @staticmethod
    def _substitute(text: str, template: Dict[str, Any]) -> str:
        """
        Substitute template variables in text.

        Handles:
        - {{key}} -> template['key']
        - {{exec(...)}} -> eval(...) for dynamic values

        Args:
            text: Text with template variables
            template: Dict of substitution values

        Returns:
            Text with substitutions applied
        """
        # First, replace simple {{key}} from template dict
        for key, value in template.items():
            text = text.replace("{{" + key + "}}", str(value))

        # Then, handle {{exec(...)}} dynamic evaluations
        while "{{exec(" in text:
            start = text.find("{{exec(")
            if start == -1:
                break

            end = text.find("}}", start)
            if end == -1:
                break

            # Extract expression
            expr = text[start + 7 : end - 1]  # Skip '{{exec(' and trailing ')'

            # Evaluate expression
            try:
                # Make common imports and stubs available in eval context
                class ContainerStub:
                    def get_config_variable(self, key, env_var, default=""):
                        return os.path.expanduser(os.getenv(env_var, default))

                class TzLocalStub:
                    @staticmethod
                    def get_localzone():
                        return "UTC"

                eval_context = {
                    "datetime": datetime,
                    "os": os,
                    "Container": lambda: ContainerStub(),  # Returns instance when called
                    "tzlocal": TzLocalStub,
                    "thread_id": "default",  # sabre doesn't use thread_id
                    "str": str,  # Make str available
                }
                result = str(eval(expr, eval_context))
            except Exception:
                # If eval fails, keep original text
                result = f"{{{{exec({expr})}}}}"

            # Replace in text
            text = text[:start] + result + text[end + 2 :]

        return text
