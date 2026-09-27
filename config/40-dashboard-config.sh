#!/bin/sh
set -eu

# nginx invokes this at container startup; no source file or backend state changes.
ui_mode="${DASHBOARD_UI_MODE:-full}"
case "$ui_mode" in
  dispatcher|full) ;;
  *) echo 'DASHBOARD_UI_MODE must be dispatcher or full' >&2; exit 1 ;;
esac
import_max_bytes="${RITM_IMPORT_MAX_BYTES:-268435456}"
import_timeout_seconds="${RITM_IMPORT_TIMEOUT_SECONDS:-120}"
for value in "$import_max_bytes" "$import_timeout_seconds"; do
  case "$value" in
    ''|*[!0-9]*) echo 'Import limits must be positive integers' >&2; exit 1 ;;
  esac
  if [ "$value" -le 0 ]; then echo 'Import limits must be positive integers' >&2; exit 1; fi
done
config_target="${1:-/usr/share/nginx/html/config.js}"
printf 'window.RITM_CONFIG = {"uiMode": "%s", "importMaxBytes": %s, "importTimeoutSeconds": %s};\n' "$ui_mode" "$import_max_bytes" "$import_timeout_seconds" > "$config_target"
