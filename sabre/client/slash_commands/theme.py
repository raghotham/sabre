"""Theme toggle command - toggle between light and dark mode"""

from .base import BaseCommand, CommandResult


class ThemeToggleCommand(BaseCommand):
    """Toggle between light and dark theme"""

    @property
    def name(self) -> str:
        return "theme-toggle"

    @property
    def description(self) -> str:
        return "Toggle between light and dark theme"

    async def execute(self, args: list) -> CommandResult:
        """Toggle the theme"""
        try:
            new_theme = self.client.tui.toggle_theme()
            return CommandResult(success=True, message=f"Theme toggled to: {new_theme}")
        except Exception as e:
            return CommandResult(success=False, message=f"Failed to toggle theme: {e}")
