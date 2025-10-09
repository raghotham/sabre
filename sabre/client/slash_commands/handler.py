"""Slash command handler for LLMVM2 Client"""
from typing import Dict
from .base import BaseCommand, CommandResult
from .help import HelpCommand
from .clear import ClearCommand
from .exit import ExitCommand, QuitCommand
from .helpers import HelpersCommand


class SlashCommandHandler:
    """Handler for slash commands in llmvm2 client"""

    def __init__(self, client):
        self.client = client

        # Initialize all commands
        self.command_instances: Dict[str, BaseCommand] = {}
        self._register_commands()

    def _register_commands(self):
        """Register all available commands"""
        commands = [
            HelpCommand(self.client),
            ClearCommand(self.client),
            ExitCommand(self.client),
            QuitCommand(self.client),
            HelpersCommand(self.client),
        ]

        for cmd in commands:
            self.command_instances[cmd.name] = cmd

    def is_slash_command(self, user_input: str) -> bool:
        """Check if input is a slash command"""
        return user_input.strip().startswith('/')

    async def execute_command(self, user_input: str) -> CommandResult:
        """Execute a slash command"""
        if not self.is_slash_command(user_input):
            return CommandResult(success=False, message="Not a slash command")

        # Parse command and arguments
        parts = user_input.strip()[1:].split()  # Remove leading '/' and split
        if not parts:
            return CommandResult(success=False, message="Empty command")

        command_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        # Check if command exists
        if command_name not in self.command_instances:
            available = ', '.join(f'/{name}' for name in self.command_instances.keys())
            return CommandResult(
                success=False,
                message=f"Unknown command '/{command_name}'. Available commands: {available}"
            )

        # Execute the command
        try:
            command = self.command_instances[command_name]
            return await command.execute(args)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Command error: {str(e)}"
            )

    def get_available_commands(self) -> list:
        """Get list of available command names"""
        return list(self.command_instances.keys())

    def get_command_help(self) -> str:
        """Get help text for all commands"""
        help_lines = ["Available slash commands:\n"]
        for name, cmd in self.command_instances.items():
            help_lines.append(f"/{name:<15} - {cmd.description}")
        return "\n".join(help_lines)
