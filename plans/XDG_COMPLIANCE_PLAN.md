# XDG Standards Compliance Plan for SABRE

## Current State Analysis

### Existing Directory Usage

**Found in codebase:**
1. **Files directory**: `~/.sabre/files/{conversation_id}/` - For matplotlib images, generated files
2. **Logs directory**: `./logs/` (project-relative) - Server/client logs
3. **API key**: `~/.openai/key` - OpenAI API key storage (not changed, external convention)

### XDG Base Directory Specification

The XDG Base Directory Specification defines:

- **`XDG_DATA_HOME`** (default: `~/.local/share/`) - User-specific data files
- **`XDG_CONFIG_HOME`** (default: `~/.config/`) - User-specific configuration
- **`XDG_STATE_HOME`** (default: `~/.local/state/`) - User-specific state data (logs, history)
- **`XDG_CACHE_HOME`** (default: `~/.cache/`) - User-specific non-essential cached files

## Proposed Directory Structure

### Standard Mode (Production)

```
~/.local/
├── state/sabre/              # XDG_STATE_HOME/sabre
│   └── logs/                 # Application logs
│       ├── server.log
│       ├── client.log
│       └── server.pid
├── share/sabre/              # XDG_DATA_HOME/sabre
│   ├── files/                # Generated files (images, downloads)
│   │   └── {conversation_id}/
│   │       ├── figure_1.png
│   │       └── ...
│   ├── memory/               # Future: persistent memory storage
│   └── conversations/        # Future: saved conversations
└── cache/sabre/              # XDG_CACHE_HOME/sabre
    └── downloads/            # Future: temporary downloads

~/.config/sabre/              # XDG_CONFIG_HOME/sabre
├── config.yaml               # User configuration
└── personas.yaml             # Custom persona definitions
```

### Debug Mode (`SABRE_HOME=./local`)

When `SABRE_HOME` is set, everything goes under that directory:

```
./local/
├── state/
│   └── logs/
│       ├── server.log
│       └── client.log
├── share/
│   └── files/
│       └── {conversation_id}/
├── cache/
│   └── downloads/
└── config/
    ├── config.yaml
    └── personas.yaml
```

## Implementation Plan

### 1. Create Paths Utility Module

**File:** `sabre/common/paths.py`

```python
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
from pathlib import Path


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
    def get_config_file() -> Path:
        """Get main config file path."""
        return SabrePaths.get_config_home() / "config.yaml"

    @staticmethod
    def get_personas_file() -> Path:
        """Get personas config file path."""
        return SabrePaths.get_config_home() / "personas.yaml"

    @staticmethod
    def get_pid_file() -> Path:
        """Get server PID file path (state)."""
        return SabrePaths.get_state_home() / "logs" / "server.pid"

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
        ]

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


# Convenience functions for common operations


def get_logs_dir() -> Path:
    """Get logs directory."""
    return SabrePaths.get_logs_dir()


def get_files_dir(conversation_id: str = None) -> Path:
    """Get files directory."""
    return SabrePaths.get_files_dir(conversation_id)


def get_config_file() -> Path:
    """Get config file path."""
    return SabrePaths.get_config_file()


def get_pid_file() -> Path:
    """Get PID file path."""
    return SabrePaths.get_pid_file()


def ensure_dirs():
    """Ensure all SABRE directories exist."""
    SabrePaths.ensure_dirs()
```

### 2. Update Code to Use Paths Module

**Files to update:**

1. **`sabre/cli.py`**
   ```python
   from sabre.common.paths import get_logs_dir, get_pid_file, ensure_dirs


   def get_log_dir():
       """Get logs directory path"""
       return get_logs_dir()


   def get_pid_file():
       """Get PID file path"""
       return get_pid_file()


   def start_server():
       """Start the SABRE server in background"""
       # Ensure directories exist
       ensure_dirs()

       # ... rest of code
   ```

2. **`sabre/server/__main__.py`**
   ```python
   from sabre.common.paths import get_logs_dir, ensure_dirs


   def main():
       """Run the sabre server."""
       # Ensure directories exist
       ensure_dirs()

       log_dir = get_logs_dir()
       # ... rest of code
   ```

3. **`sabre/server/api/server.py`**
   ```python
   from sabre.common.paths import get_logs_dir, get_files_dir, ensure_dirs


   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # Ensure directories exist
       ensure_dirs()

       log_dir = get_logs_dir()
       # ... rest of code


   @app.get("/files/{conversation_id}/{filename}")
   async def serve_file(conversation_id: str, filename: str):
       """Serve files generated during conversation."""
       files_dir = get_files_dir(conversation_id)
       file_path = files_dir / filename
       # ... rest of code
   ```

4. **`sabre/server/orchestrator.py`**
   ```python
   from sabre.common.paths import get_files_dir


   async def _save_image_to_disk(
       self, image_content: ImageContent, conversation_id: str, filename: str
   ) -> str:
       """Save image to disk and return URL."""
       from pathlib import Path
       import base64

       # Get files directory for this conversation
       files_dir = get_files_dir(conversation_id)
       files_dir.mkdir(parents=True, exist_ok=True)

       # Decode and save image
       image_bytes = base64.b64decode(image_content.image_data)
       file_path = files_dir / filename
       file_path.write_bytes(image_bytes)

       # Generate URL (assumes server on localhost:8011)
       port = os.getenv("LLMVM_PORT", "8011")
       return f"http://localhost:{port}/files/{conversation_id}/{filename}"
   ```

5. **`sabre/client/client.py`**
   ```python
   from sabre.common.paths import get_logs_dir, ensure_dirs

   # In logging setup
   ensure_dirs()
   log_dir = get_logs_dir()
   # ... rest of code
   ```

### 3. Migration Strategy

**Backward Compatibility:**

On first run, check if old directories exist and migrate:

```python
# In sabre/common/paths.py


@staticmethod
def migrate_from_old_structure():
    """Migrate from old ~/.sabre structure to XDG-compliant structure."""
    old_base = Path.home() / ".sabre"

    if not old_base.exists():
        return  # Nothing to migrate

    import shutil
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Migrating from old directory structure...")

    # Ensure new directories exist
    SabrePaths.ensure_dirs()

    # Migrate files
    old_files = old_base / "files"
    if old_files.exists():
        new_files = SabrePaths.get_files_dir()
        logger.info(f"Migrating files: {old_files} -> {new_files}")
        for item in old_files.iterdir():
            shutil.move(str(item), str(new_files / item.name))

    # Remove old directory if empty
    try:
        old_base.rmdir()
        logger.info(f"Removed old directory: {old_base}")
    except OSError:
        logger.warning(f"Could not remove old directory (not empty): {old_base}")
```

Call this in startup:
```python
# In sabre/server/__main__.py and sabre/cli.py
from sabre.common.paths import ensure_dirs, migrate_from_old_structure


def main():
    migrate_from_old_structure()
    ensure_dirs()
    # ... rest of code
```

### 4. Environment Variable Support

**`SABRE_HOME` for debugging:**

```bash
# Debug mode - everything goes to ./local
export SABRE_HOME=./local
uv run sabre

# Result:
./local/
├── state/logs/
├── share/files/
├── config/
└── cache/
```

**Standard XDG variables also work:**

```bash
# Override just data directory
export XDG_DATA_HOME=/tmp/sabre-test/data
uv run sabre

# Result:
/tmp/sabre-test/data/sabre/files/
~/.local/state/sabre/logs/  # Other dirs still in default location
```

### 5. Documentation Updates

**Add to README.md:**

```markdown
## Directory Structure

SABRE follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html):

### Standard Locations

- **Logs**: `~/.local/state/sabre/logs/`
- **Generated files**: `~/.local/share/sabre/files/`
- **Configuration**: `~/.config/sabre/`
- **Cache**: `~/.cache/sabre/`

### Debug Mode

For development, set `SABRE_HOME` to use a local directory:

```bash
export SABRE_HOME=./local
uv run sabre
```

All SABRE data will be stored in `./local/` instead of `~/.local/`.

### Custom XDG Paths

SABRE respects standard XDG environment variables:

- `XDG_DATA_HOME` (default: `~/.local/share`)
- `XDG_CONFIG_HOME` (default: `~/.config`)
- `XDG_STATE_HOME` (default: `~/.local/state`)
- `XDG_CACHE_HOME` (default: `~/.cache`)
```

## Implementation Checklist

- [ ] Create `sabre/common/paths.py` with `SabrePaths` class
- [ ] Update `sabre/cli.py` to use paths module
- [ ] Update `sabre/server/__main__.py` to use paths module
- [ ] Update `sabre/server/api/server.py` to use paths module
- [ ] Update `sabre/server/orchestrator.py` to use paths module
- [ ] Update `sabre/client/client.py` to use paths module
- [ ] Add migration logic for old `~/.sabre` directory
- [ ] Test with `SABRE_HOME=./local`
- [ ] Test with custom XDG variables
- [ ] Update README.md with directory documentation
- [ ] Update `.gitignore` to exclude `./local/`

## Testing

### Test Cases

1. **Fresh install (no existing dirs)**
   ```bash
   rm -rf ~/.local/state/sabre ~/.local/share/sabre ~/.config/sabre
   uv run sabre
   # Verify directories created correctly
   ```

2. **Migration from old structure**
   ```bash
   mkdir -p ~/.sabre/files/test_conv
   touch ~/.sabre/files/test_conv/test.png
   uv run sabre
   # Verify file migrated to ~/.local/share/sabre/files/test_conv/test.png
   ```

3. **Debug mode**
   ```bash
   export SABRE_HOME=./local
   uv run sabre
   # Verify everything in ./local/
   ls -la ./local/state/logs/
   ls -la ./local/share/files/
   ```

4. **Custom XDG variables**
   ```bash
   export XDG_STATE_HOME=/tmp/test-state
   export XDG_DATA_HOME=/tmp/test-data
   uv run sabre
   # Verify logs in /tmp/test-state/sabre/
   # Verify files in /tmp/test-data/sabre/
   ```

## Benefits

✅ **Standards Compliant** - Follows XDG spec like proper Unix tools
✅ **Debug Friendly** - `SABRE_HOME=./local` for isolated testing
✅ **Flexible** - Respects user's XDG preferences
✅ **Clean** - No more scattered `~/.sabre`, `./logs/` mix
✅ **Professional** - Behaves like established tools (cargo, npm, etc.)
✅ **Migration Support** - Smooth upgrade from old structure
