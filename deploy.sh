#!/usr/bin/env bash
# deploy.sh — Deploy osmose-python to the production server
#
# Runs OSMOSE as a standalone Uvicorn service (osmose-shiny.service) on port
# 8838, proxied by nginx.  This bypasses shiny-server which has WebSocket
# compatibility issues with Python Shiny + Starlette.
#
# Usage:  sudo bash deploy.sh
#         sudo bash deploy.sh --uninstall
#         sudo bash deploy.sh --restart      # restart the service only

set -euo pipefail

APP_NAME="osmose"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
SHINY_ROOT="/srv/shiny-server"
SHINY_PYTHON="/opt/micromamba/envs/shiny/bin/python3"
SHINY_PIP="/opt/micromamba/envs/shiny/bin/pip"
LINK_PATH="${SHINY_ROOT}/${APP_NAME}"
SERVICE_NAME="osmose-shiny"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
APP_PORT=8838

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*" >&2; }

# --- Restart mode ---
if [[ "${1:-}" == "--restart" ]]; then
    info "Restarting ${SERVICE_NAME}..."
    systemctl restart "$SERVICE_NAME" 2>/dev/null && info "Service restarted." || error "Could not restart ${SERVICE_NAME}"
    exit 0
fi

# --- Uninstall mode ---
if [[ "${1:-}" == "--uninstall" ]]; then
    info "Uninstalling ${APP_NAME}..."

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl stop "$SERVICE_NAME"
        info "Stopped ${SERVICE_NAME} service"
    fi

    if [[ -f "$SERVICE_FILE" ]]; then
        systemctl disable "$SERVICE_NAME" 2>/dev/null || true
        rm "$SERVICE_FILE"
        systemctl daemon-reload
        info "Removed ${SERVICE_FILE}"
    fi

    if [[ -L "$LINK_PATH" ]]; then
        rm "$LINK_PATH"
        info "Removed symlink ${LINK_PATH}"
    else
        warn "No symlink at ${LINK_PATH}"
    fi

    info "Uninstall complete."
    info "NOTE: Update nginx config manually to remove the /osmose/ location block."
    exit 0
fi

# --- Pre-flight checks ---
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (use sudo)."
    exit 1
fi

if [[ ! -f "${SOURCE_DIR}/app.py" ]]; then
    error "app.py not found in ${SOURCE_DIR}. Run this script from the project root."
    exit 1
fi

if [[ ! -d "$SHINY_ROOT" ]]; then
    error "Shiny server directory ${SHINY_ROOT} not found."
    exit 1
fi

if [[ ! -f "$SHINY_PYTHON" ]]; then
    error "Shiny Python not found at ${SHINY_PYTHON}."
    exit 1
fi

# --- Step 1: Create symlink ---
if [[ -L "$LINK_PATH" ]]; then
    current_target="$(readlink "$LINK_PATH")"
    if [[ "$current_target" == "$SOURCE_DIR" ]]; then
        info "Symlink already exists: ${LINK_PATH} -> ${SOURCE_DIR}"
    else
        warn "Symlink exists but points to ${current_target}. Updating..."
        rm "$LINK_PATH"
        ln -s "$SOURCE_DIR" "$LINK_PATH"
        info "Updated symlink: ${LINK_PATH} -> ${SOURCE_DIR}"
    fi
elif [[ -e "$LINK_PATH" ]]; then
    error "${LINK_PATH} exists and is not a symlink. Remove it manually."
    exit 1
else
    ln -s "$SOURCE_DIR" "$LINK_PATH"
    info "Created symlink: ${LINK_PATH} -> ${SOURCE_DIR}"
fi

# Ensure shiny user can read the source directory
chown -h shiny:shiny "$LINK_PATH" 2>/dev/null || true

# --- Step 2: Install missing Python dependencies ---
MISSING_PKGS=()
for pkg in pymoo SALib; do
    if ! "$SHINY_PIP" show "$pkg" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing missing packages: ${MISSING_PKGS[*]}"
    "$SHINY_PIP" install "${MISSING_PKGS[@]}" --quiet
    info "Packages installed."
else
    info "All Python dependencies already installed."
fi

# --- Step 3: Install systemd service ---
info "Installing ${SERVICE_NAME} systemd service..."
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=OSMOSE Python Shiny App (direct Uvicorn)
After=network.target

[Service]
Type=simple
User=shiny
Group=shiny
WorkingDirectory=${LINK_PATH}
ExecStart=${SHINY_PYTHON} -m uvicorn app:app --host 127.0.0.1 --port ${APP_PORT} --root-path /${APP_NAME}
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME" 2>/dev/null

# --- Step 4: Start/restart the service ---
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl restart "$SERVICE_NAME"
    info "Service ${SERVICE_NAME} restarted."
else
    systemctl start "$SERVICE_NAME"
    info "Service ${SERVICE_NAME} started."
fi

# Wait for it to be ready
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    info "Service is running."
else
    error "Service failed to start. Check: journalctl -u ${SERVICE_NAME} --no-pager -n 20"
    exit 1
fi

# --- Step 5: Verify HTTP ---
HTTP_CODE=$(curl -sS -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${APP_PORT}/" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    info "HTTP check passed (port ${APP_PORT})."
else
    warn "HTTP check returned ${HTTP_CODE}. Service may still be starting."
fi

# --- Summary ---
echo ""
info "Deployment complete!"
echo "  Service:    ${SERVICE_NAME} (port ${APP_PORT})"
echo "  Source:     ${SOURCE_DIR}"
echo "  Symlink:    ${LINK_PATH}"
echo "  Service:    ${SERVICE_FILE}"
echo ""
echo "  NOTE: Ensure nginx proxies /osmose/ to http://127.0.0.1:${APP_PORT}/"
echo "        (not to shiny-server on port 3838)"
echo ""
echo "  Commands:"
echo "    Restart:    sudo bash ${SOURCE_DIR}/deploy.sh --restart"
echo "    Logs:       journalctl -u ${SERVICE_NAME} -f"
echo "    Uninstall:  sudo bash ${SOURCE_DIR}/deploy.sh --uninstall"
