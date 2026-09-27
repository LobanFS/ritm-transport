#!/bin/sh
set -eu

# nginx invokes this at container startup; no source file or backend state changes.
ui_mode="${DASHBOARD_UI_MODE:-full}"
case "$ui_mode" in
  dispatcher|full) ;;
  *) echo 'DASHBOARD_UI_MODE must be dispatcher or full' >&2; exit 1 ;;
esac
config_target="${1:-/usr/share/nginx/html/config.js}"
printf 'window.RITM_CONFIG = {"uiMode": "%s"};\n' "$ui_mode" > "$config_target"
