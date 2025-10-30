"""Theme command - toggle between light and dark mode"""

from .base import BaseCommand, CommandResult


class ThemeCommand(BaseCommand):
    """Toggle between light and dark theme"""

    @property
    def name(self) -> str:
        return "theme"

    @property
    def description(self) -> str:
        return "Toggle between light and dark theme"

    async def execute(self, args: list) -> CommandResult:
        """Toggle the theme or set to specific value"""
        # If args provided, try to set specific theme
        if args:
            theme = args[0].lower()
            if theme in ("light", "dark"):
                try:
                    self.client.tui.set_theme(theme)
                    return CommandResult(success=True, message=f"Theme set to: {theme}")
                except Exception as e:
                    return CommandResult(success=False, message=f"Failed to set theme: {e}")
            else:
                return CommandResult(
                    success=False, message=f"Invalid theme: {theme}. Use 'light' or 'dark', or omit to toggle."
                )
        else:
            # No args - toggle theme
            try:
                new_theme = self.client.tui.toggle_theme()
                return CommandResult(success=True, message=f"Theme toggled to: {new_theme}")
            except Exception as e:
                return CommandResult(success=False, message=f"Failed to toggle theme: {e}")
