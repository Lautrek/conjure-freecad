#!/bin/bash
# Conjure MCP Server Launcher
#
# This script launches the MCP server from the installed workbench location.
# It is bundled with the FreeCAD workbench at:
#   ~/.local/share/FreeCAD/Mod/conjure/mcp_server.sh
#
# Usage in .mcp.json (Claude Code / Cursor / etc.):
#   {
#     "mcpServers": {
#       "conjure": {
#         "type": "stdio",
#         "command": "~/.local/share/FreeCAD/Mod/conjure/mcp_server.sh",
#         "env": {
#           "CONJURE_API_KEY": "<your-api-key>"
#         }
#       }
#     }
#   }

LOG_DIR="$HOME/.conjure/logs"
LOG_FILE="$LOG_DIR/mcp_shell_debug.log"
mkdir -p "$LOG_DIR"

# Discover install location (where this script lives)
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source user config if it exists
if [ -f "$HOME/.conjure/.env" ]; then
    set -a
    source "$HOME/.conjure/.env"
    set +a
fi

# Set PYTHONPATH to include bundled bridge and client modules
# The install copies bridge/ and src/ into the install directory
export PYTHONPATH="$INSTALL_DIR/src:$INSTALL_DIR/bridge_root:$PYTHONPATH"

echo "=== MCP Server (installed) ===" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "INSTALL_DIR: $INSTALL_DIR" >> "$LOG_FILE"
echo "PYTHONPATH: $PYTHONPATH" >> "$LOG_FILE"

# Change to the bridge_root directory so `python3 -m bridge.conjure_bridge` works
cd "$INSTALL_DIR/bridge_root"

# Run the MCP server
exec python3 -m bridge.conjure_bridge 2>> "$LOG_FILE"
