"""
XDG-compliant directory paths for SABRE.

Follows XDG Base Directory Specification:
https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

Environment variables:
- SABRE_HOME: Override base directory for debugging (default: ~/.local)
- XDG_DATA_HOME: User data directory (default: ~/.local/share)
- XDG_CONFIG_HOME: User config directory (default: ~/.config)
- XDG_STATE_HOME: User state directory (default: ~/.local/state)
- XDG_CACHE_HOME: User cache directory (default: ~/.cache)
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SabrePaths:
    """Central location for all SABRE directory paths."""

    @staticmethod
    def get_base_dir() -> Path:
        """
        Get base directory for SABRE.

        Returns SABRE_HOME if set (for debugging), otherwise ~/.local
        """
        sabre_home = os.getenv("SABRE_HOME")
        if sabre_home:
            return Path(sabre_home).expanduser().resolve()
        return Path.home() / ".local"

    @staticmethod
    def get_data_home() -> Path:
        """Get XDG_DATA_HOME/sabre directory."""
        # Check for debug override
        sabre_home = os.getenv("SABRE_HOME")
        if sabre_home:
            return Path(sabre_home).expanduser() / "share"

        # Use XDG_DATA_HOME or default
        xdg_data = os.getenv("XDG_DATA_HOME")
        if xdg_data:
            return Path(xdg_data) / "sabre"

        return Path.home() / ".local" / "share" / "sabre"

    @staticmethod
    def get_config_home() -> Path:
        """Get XDG_CONFIG_HOME/sabre directory."""
        # Check for debug override
        sabre_home = os.getenv("SABRE_HOME")
        if sabre_home:
            return Path(sabre_home).expanduser() / "config"

        # Use XDG_CONFIG_HOME or default
        xdg_config = os.getenv("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / "sabre"

        return Path.home() / ".config" / "sabre"

    @staticmethod
    def get_state_home() -> Path:
        """Get XDG_STATE_HOME/sabre directory."""
        # Check for debug override
        sabre_home = os.getenv("SABRE_HOME")
        if sabre_home:
            return Path(sabre_home).expanduser() / "state"

        # Use XDG_STATE_HOME or default
        xdg_state = os.getenv("XDG_STATE_HOME")
        if xdg_state:
            return Path(xdg_state) / "sabre"

        return Path.home() / ".local" / "state" / "sabre"

    @staticmethod
    def get_cache_home() -> Path:
        """Get XDG_CACHE_HOME/sabre directory."""
        # Check for debug override
        sabre_home = os.getenv("SABRE_HOME")
        if sabre_home:
            return Path(sabre_home).expanduser() / "cache"

        # Use XDG_CACHE_HOME or default
        xdg_cache = os.getenv("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / "sabre"

        return Path.home() / ".cache" / "sabre"

    # Convenience methods for specific paths

    @staticmethod
    def get_logs_dir() -> Path:
        """Get logs directory (state)."""
        return SabrePaths.get_state_home() / "logs"

    @staticmethod
    def get_files_dir(conversation_id: str = None) -> Path:
        """
        Get files directory for generated content (data).

        Args:
            conversation_id: Optional conversation ID for subdirectory

        Returns:
            Path to files directory
        """
        base = SabrePaths.get_data_home() / "files"
        if conversation_id:
            return base / conversation_id
        return base

    @staticmethod
    def get_pid_file() -> Path:
        """Get server PID file path (state)."""
        return SabrePaths.get_state_home() / "logs" / "server.pid"

    # Session-based directory methods

    @staticmethod
    def get_sessions_base_dir() -> Path:
        """Get base directory for all sessions (state)."""
        return SabrePaths.get_logs_dir() / "sessions"

    @staticmethod
    def get_session_dir(session_id: str) -> Path:
        """
        Get session directory containing log and files.

        Args:
            session_id: Session ID

        Returns:
            Path to session directory (~/.local/state/sabre/logs/sessions/{session_id})
        """
        return SabrePaths.get_sessions_base_dir() / session_id

    @staticmethod
    def get_session_log_file(session_id: str) -> Path:
        """
        Get session log file path.

        Args:
            session_id: Session ID

        Returns:
            Path to session.jsonl file
        """
        return SabrePaths.get_session_dir(session_id) / "session.jsonl"

    @staticmethod
    def get_session_files_dir(session_id: str) -> Path:
        """
        Get session files directory for generated content (images, etc).

        Args:
            session_id: Session ID

        Returns:
            Path to files directory within session
        """
        return SabrePaths.get_session_dir(session_id) / "files"

    @staticmethod
    def get_session_workspace_dir(session_id: str) -> Path:
        """
        Get session workspace directory for Bash execution.

        This directory is used as the working directory for all Bash.execute()
        commands in the session. It can be mounted into containers to share
        the filesystem between host and container.

        Args:
            session_id: Session ID

        Returns:
            Path to workspace directory within session
        """
        return SabrePaths.get_session_dir(session_id) / "workspace"

    @staticmethod
    def ensure_dirs():
        """Create all necessary directories if they don't exist."""
        dirs = [
            SabrePaths.get_data_home(),
            SabrePaths.get_config_home(),
            SabrePaths.get_state_home(),
            SabrePaths.get_cache_home(),
            SabrePaths.get_logs_dir(),
            SabrePaths.get_files_dir(),
            SabrePaths.get_sessions_base_dir(),
        ]

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def migrate_from_old_structure():
        """Migrate from old ~/.sabre structure to XDG-compliant structure."""
        old_base = Path.home() / ".sabre"

        if not old_base.exists():
            return  # Nothing to migrate

        import shutil

        logger.info("Migrating from old directory structure...")

        # Ensure new directories exist
        SabrePaths.ensure_dirs()

        # Migrate files
        old_files = old_base / "files"
        if old_files.exists():
            new_files = SabrePaths.get_files_dir()
            logger.info(f"Migrating files: {old_files} -> {new_files}")
            for item in old_files.iterdir():
                dest = new_files / item.name
                if dest.exists():
                    logger.warning(f"Skipping existing: {dest}")
                    continue
                shutil.move(str(item), str(dest))

        # Remove old directory if empty
        try:
            if old_files.exists():
                old_files.rmdir()
            old_base.rmdir()
            logger.info(f"Removed old directory: {old_base}")
        except OSError as e:
            logger.warning(f"Could not remove old directory (not empty): {old_base} - {e}")

    @staticmethod
    def cleanup_all(force: bool = False) -> dict:
        """
        Clean up all SABRE XDG directories.

        Args:
            force: If True, skip confirmation and delete immediately

        Returns:
            Dictionary with cleanup results including:
            - directories: List of directories that would be/were removed
            - sizes: Dictionary of directory sizes in bytes
            - total_size: Total size in bytes
            - removed: Whether directories were actually removed
        """
        import shutil

        # Get all SABRE directories
        directories = [
            SabrePaths.get_data_home(),
            SabrePaths.get_config_home(),
            SabrePaths.get_state_home(),
            SabrePaths.get_cache_home(),
        ]

        # Calculate sizes
        sizes = {}
        total_size = 0
        existing_dirs = []

        for directory in directories:
            if directory.exists():
                existing_dirs.append(directory)
                # Calculate directory size
                dir_size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
                sizes[str(directory)] = dir_size
                total_size += dir_size

        result = {
            "directories": [str(d) for d in existing_dirs],
            "sizes": sizes,
            "total_size": total_size,
            "removed": False,
        }

        # If force is True, proceed with deletion
        if force and existing_dirs:
            for directory in existing_dirs:
                try:
                    shutil.rmtree(directory)
                    logger.info(f"Removed directory: {directory}")
                except Exception as e:
                    logger.error(f"Failed to remove {directory}: {e}")
                    raise

            result["removed"] = True

        return result


# Convenience functions for common operations


def get_logs_dir() -> Path:
    """Get logs directory."""
    return SabrePaths.get_logs_dir()


def get_files_dir(conversation_id: str = None) -> Path:
    """Get files directory."""
    return SabrePaths.get_files_dir(conversation_id)


def get_pid_file() -> Path:
    """Get PID file path."""
    return SabrePaths.get_pid_file()


def ensure_dirs():
    """Ensure all SABRE directories exist."""
    SabrePaths.ensure_dirs()


def migrate_from_old_structure():
    """Migrate from old directory structure."""
    SabrePaths.migrate_from_old_structure()


def cleanup_all(force: bool = False) -> dict:
    """Clean up all SABRE directories."""
    return SabrePaths.cleanup_all(force=force)


def get_sessions_base_dir() -> Path:
    """Get base directory for all sessions."""
    return SabrePaths.get_sessions_base_dir()


def get_session_dir(session_id: str) -> Path:
    """Get session directory."""
    return SabrePaths.get_session_dir(session_id)


def get_session_log_file(session_id: str) -> Path:
    """Get session log file path."""
    return SabrePaths.get_session_log_file(session_id)


def get_session_files_dir(session_id: str) -> Path:
    """Get session files directory."""
    return SabrePaths.get_session_files_dir(session_id)


def get_session_workspace_dir(session_id: str) -> Path:
    """Get session workspace directory."""
    return SabrePaths.get_session_workspace_dir(session_id)
