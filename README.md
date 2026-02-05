# Conjure FreeCAD Workbench

[![CI](https://github.com/Lautrek/conjure-freecad/actions/workflows/ci.yml/badge.svg)](https://github.com/Lautrek/conjure-freecad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

FreeCAD workbench for AI-assisted CAD with Conjure. Enables Claude Code and other AI tools to control FreeCAD via the Model Context Protocol (MCP).

## Features

- AI-assisted 3D modeling via natural language
- Full FreeCAD API access through MCP tools
- Real-time model updates and visualization
- Engineering materials library
- Works with Claude Code out of the box

## Requirements

- FreeCAD 1.0+
- Python 3.10+

## Installation

### Option 1: FreeCAD Addon Manager (Recommended)

1. Open FreeCAD
2. Go to **Edit > Preferences > Addon Manager**
3. Search for "Conjure"
4. Click **Install**
5. Restart FreeCAD

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/Lautrek/conjure-freecad.git

# Copy workbench to FreeCAD modules directory
cp -r conjure-freecad/workbench ~/.local/share/FreeCAD/Mod/conjure

# Restart FreeCAD
```

### Option 3: Using the build script

```bash
git clone https://github.com/Lautrek/conjure-freecad.git
cd conjure-freecad
./build.sh
# Install the generated ZIP via FreeCAD Addon Manager
```

## Quick Start

1. **Start FreeCAD** with the Conjure workbench installed
2. **Configure Claude Code** to use the Conjure MCP server:

Edit `~/.claude/workspace-settings.json`:
```json
{
  "mcpServers": {
    "conjure": {
      "command": "freecad",
      "args": ["--run-script", "path/to/mcp_server.sh"]
    }
  }
}
```

3. **Start designing** with natural language:
```
"Create a box 100x50x30mm and add a 10mm fillet on the top edges"
```

## Configuration

Create `~/.conjure/config.yaml`:

```yaml
# FreeCAD connection
freecad:
  host: localhost
  port: 9876

# Optional: Connect to Conjure cloud server
server:
  enabled: false
  url: https://api.conjure.lautrek.com
  api_key: sk-your-api-key
```

## Architecture

```
Claude Code (Editor)
    | (MCP stdio)
Conjure Workbench (FreeCAD)
    |-> CAD Operations (local)
    \-> (Optional) Cloud Server
```

The workbench:
- Receives MCP tool calls from Claude Code
- Executes operations directly in FreeCAD
- Optionally syncs with cloud server for collaboration

## Development

```bash
# Clone the repo
git clone https://github.com/Lautrek/conjure-freecad.git
cd conjure-freecad

# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check workbench/ src/

# Build distribution ZIP
./build.sh
```

## Project Structure

```
conjure-freecad/
├── workbench/           # FreeCAD workbench (installed to ~/.local/share/FreeCAD/Mod/)
│   ├── InitGui.py       # Workbench UI initialization
│   ├── Init.py          # Module initialization
│   ├── conjure.py       # Core workbench commands
│   ├── logger.py        # Logging utilities
│   ├── assets/          # Icons and images
│   └── shared/          # Shared utilities
├── src/                 # Python package
│   └── adapter/         # FreeCAD adapter for Conjure SDK
├── build.sh             # Build script for distribution ZIP
└── config.example.yaml  # Example configuration
```

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

MIT License - See [LICENSE](LICENSE) file.

## Links

- [Conjure SDK](https://github.com/Lautrek/conjure-sdk) - Python SDK (`pip install conjure-sdk`)
- [Conjure Blender](https://github.com/Lautrek/conjure-blender) - Blender add-on
- [Conjure Website](https://conjure.lautrek.com)
- [Issues](https://github.com/Lautrek/conjure-freecad/issues)
