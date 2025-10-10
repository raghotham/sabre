"""Clear command - clears conversation context"""

from .base import BaseCommand, CommandResult


class ClearCommand(BaseCommand):
    """Clear conversation context"""

    @property
    def name(self) -> str:
        return "clear"

    @property
    def description(self) -> str:
        return "Clear conversation context and start a new conversation"

    async def execute(self, args: list) -> CommandResult:
        """Clear the conversation context by signaling server to create new conversation"""
        return CommandResult(success=True, message="Clearing conversation...", data={"action": "clear_conversation"})
