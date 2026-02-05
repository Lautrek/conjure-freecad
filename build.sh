#!/bin/bash
# Build FreeCAD workbench distribution ZIP
set -e

cd "$(dirname "$0")"

# Extract version from pyproject.toml
VERSION=$(grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2)
echo "Building conjure-freecad v${VERSION}..."

# Create dist directory
mkdir -p dist

# Create a temporary build directory with proper structure
# FreeCAD expects: Mod/conjure/InitGui.py
BUILD_DIR=$(mktemp -d)
CONJURE_DIR="$BUILD_DIR/conjure"
mkdir -p "$CONJURE_DIR"

# Copy workbench files to conjure/ root
cp -r workbench/* "$CONJURE_DIR/"

# Copy src directory
cp -r src "$CONJURE_DIR/"

# Copy config example
cp config.example.yaml "$CONJURE_DIR/"

# Bundle MCP bridge for standalone MCP server
mkdir -p "$CONJURE_DIR/bridge_root/bridge"
cp ../../server/src/bridge/conjure_bridge.py "$CONJURE_DIR/bridge_root/bridge/"
cp ../../server/src/bridge/mcp_auth.py       "$CONJURE_DIR/bridge_root/bridge/"
cp ../../server/src/bridge/__init__.py       "$CONJURE_DIR/bridge_root/bridge/"
cp -r ../../server/src/bridge/resources      "$CONJURE_DIR/bridge_root/bridge/"
cp -r ../../server/src/bridge/prompts        "$CONJURE_DIR/bridge_root/bridge/"

# Create ZIP with conjure folder at root
ZIPFILE="dist/conjure-freecad-${VERSION}.zip"
rm -f "$ZIPFILE"

cd "$BUILD_DIR"
zip -r "$OLDPWD/$ZIPFILE" conjure -x "*.pyc" -x "*/__pycache__/*" -x "*.DS_Store"

# Clean up
rm -rf "$BUILD_DIR"

cd "$OLDPWD"

# Create a consistent filename for upload scripts
cp "$ZIPFILE" "conjure_freecad_client.zip"

echo "Built: $ZIPFILE"
echo "Also: conjure_freecad_client.zip"
ls -lh "$ZIPFILE" conjure_freecad_client.zip

# Verify structure
echo ""
echo "ZIP contents:"
unzip -l "$ZIPFILE" | head -20
