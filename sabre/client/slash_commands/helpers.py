"""Helpers command - shows available Python helpers"""
from .base import BaseCommand, CommandResult


class HelpersCommand(BaseCommand):
    """Show available Python helpers and functions"""

    @property
    def name(self) -> str:
        return "helpers"

    @property
    def description(self) -> str:
        return "Show available Python helpers and functions"

    async def execute(self, args: list) -> CommandResult:
        """Request helpers list from server"""
        # Return a special action to send to server
        return CommandResult(
            success=True,
            message="Requesting helpers from server...",
            data={"action": "request_helpers"}
        )
