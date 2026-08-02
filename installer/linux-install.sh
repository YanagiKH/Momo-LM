#!/usr/bin/env sh
set -eu

INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/momo-lm"
BIN_ROOT="${XDG_BIN_HOME:-$HOME/.local/bin}"
SCRIPT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

mkdir -p "$INSTALL_ROOT" "$BIN_ROOT"
cp -R "$SCRIPT_ROOT/app/." "$INSTALL_ROOT/"
chmod +x "$INSTALL_ROOT/Momo-LM"
ln -sf "$INSTALL_ROOT/Momo-LM" "$BIN_ROOT/momo-lm"

printf '%s\n' "Momo-LM installed to $INSTALL_ROOT"
printf '%s\n' "Start with: $BIN_ROOT/momo-lm"
case ":$PATH:" in
  *":$BIN_ROOT:"*) ;;
  *) printf '%s\n' "Add $BIN_ROOT to PATH to run momo-lm from any terminal." ;;
esac
