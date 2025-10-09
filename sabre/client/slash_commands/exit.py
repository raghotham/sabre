"""Exit command - exits the client"""
from .base import BaseCommand, CommandResult


class ExitCommand(BaseCommand):
    """Exit the client"""

    @property
    def name(self) -> str:
        return "exit"

    @property
    def description(self) -> str:
        return "Exit the client"

    async def execute(self, args: list) -> CommandResult:
        """Exit the client"""
        return CommandResult(
            success=True,
            message="exit",  # Special marker for client to exit
            data={"action": "exit"}
        )


class QuitCommand(ExitCommand):
    """Quit the client (alias for exit)"""

    @property
    def name(self) -> str:
        return "quit"

    @property
    def description(self) -> str:
        return "Exit the client (alias for /exit)"
