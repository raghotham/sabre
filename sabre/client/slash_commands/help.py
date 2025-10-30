"""Help command - shows available commands"""

from .base import BaseCommand, CommandResult


class HelpCommand(BaseCommand):
    """Show available slash commands"""

    @property
    def name(self) -> str:
        return "help"

    @property
    def description(self) -> str:
        return "Show this help message"

    async def execute(self, args: list) -> CommandResult:
        """Show help for all available commands"""
        help_text = """Available slash commands:

/help           - Show this help message
/helpers        - Show available Python helpers and functions
/theme [mode]   - Toggle theme or set to 'light' or 'dark'
/clear          - Note about conversation context
/exit, /quit    - Exit the client

Slash commands are processed locally and don't require server round-trips."""

        return CommandResult(success=True, message=help_text)
