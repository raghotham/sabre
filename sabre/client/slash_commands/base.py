"""Base classes for slash commands"""

from dataclasses import dataclass
from typing import Optional, Any, Dict


@dataclass
class CommandResult:
    """Result of executing a slash command"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class BaseCommand:
    """Base class for slash commands"""

    def __init__(self, client):
        self.client = client

    async def execute(self, args: list) -> CommandResult:
        """Execute the command with given arguments"""
        raise NotImplementedError("Subclasses must implement execute()")

    @property
    def name(self) -> str:
        """Command name (without the /)"""
        raise NotImplementedError("Subclasses must implement name property")

    @property
    def description(self) -> str:
        """Short description of the command"""
        raise NotImplementedError("Subclasses must implement description property")
