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

# Prod runs from its OWN git clone (PROD_SRC), NOT a symlink to the live dev tree.
# Rationale: a long-running uvicorn process reads its source via inspect
# (Shiny 1.6.3's per-renderer OTel `extract_source_ref` → `inspect.getsourcelines`).
# If the source file is edited under the running process — which is exactly what a
# symlink to the dev working tree allows — recorded line numbers drift from disk and
# `getsourcelines` raises `tokenize.TokenError`, aborting `server()` and silently
# breaking every interactive handler until restart. A dedicated clone only changes
# on an explicit deploy (which restarts), so dev edits never touch prod.
REPO_URL="${OSMOSE_REPO_URL:-https://github.com/razinkele/osmopy.git}"  # public → no creds
DEPLOY_REF="${OSMOSE_DEPLOY_REF:-origin/master}"  # what to ship; override to pin a sha/tag
PROD_SRC="${SHINY_ROOT}/${APP_NAME}-src"  # the prod clone (real dir; LINK_PATH → here)

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

    if [[ -d "$PROD_SRC" ]]; then
        rm -rf "$PROD_SRC"
        info "Removed prod clone ${PROD_SRC}"
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

if ! command -v git >/dev/null 2>&1; then
    error "git not found; required to manage the prod source clone (${PROD_SRC})."
    exit 1
fi

# --- Step 1: Sync the prod source clone ---
# Clone once, then fetch + checkout the target ref on every deploy. This is the
# ONLY moment prod's source changes (followed by a restart in Step 4), so the
# running process never reads a file that is being edited underneath it.
git config --global --add safe.directory "$PROD_SRC" 2>/dev/null || true
if [[ -d "${PROD_SRC}/.git" ]]; then
    info "Updating prod clone at ${PROD_SRC} (fetch ${DEPLOY_REF})..."
    git -C "$PROD_SRC" fetch --quiet --prune origin
elif [[ -e "$PROD_SRC" ]]; then
    error "${PROD_SRC} exists but is not a git clone. Remove it manually."
    exit 1
else
    info "Creating prod clone: ${REPO_URL} -> ${PROD_SRC}"
    git clone --quiet "$REPO_URL" "$PROD_SRC"
fi
git -C "$PROD_SRC" checkout --quiet --force --detach "$DEPLOY_REF"
DEPLOYED_SHA="$(git -C "$PROD_SRC" rev-parse --short HEAD)"
info "Prod source at ${DEPLOY_REF} (${DEPLOYED_SHA})."

# The osmose-java JAR is NOT tracked in git (so it is absent from a fresh clone).
# Provision it from the deploy host's tree so Java-engine runs remain available.
# Best-effort: prod's primary path is the Python engine; missing JAR only disables
# Java runs (the run page shows "no JAR found").
if compgen -G "${SOURCE_DIR}/osmose-java/"*.jar >/dev/null 2>&1; then
    mkdir -p "${PROD_SRC}/osmose-java"
    cp -f "${SOURCE_DIR}/osmose-java/"*.jar "${PROD_SRC}/osmose-java/"
    info "Provisioned Java JAR(s) into ${PROD_SRC}/osmose-java/."
else
    warn "No osmose-java/*.jar in ${SOURCE_DIR}; Java-engine runs will be unavailable in prod."
fi

# shiny must read the whole clone.
chown -R shiny:shiny "$PROD_SRC" 2>/dev/null || true

# --- Step 1b: Point the served path at the clone (NOT the dev tree) ---
if [[ -L "$LINK_PATH" ]]; then
    current_target="$(readlink "$LINK_PATH")"
    if [[ "$current_target" == "$PROD_SRC" ]]; then
        info "Symlink already correct: ${LINK_PATH} -> ${PROD_SRC}"
    else
        warn "Symlink points to ${current_target}; repointing to the prod clone..."
        rm "$LINK_PATH"
        ln -s "$PROD_SRC" "$LINK_PATH"
        info "Updated symlink: ${LINK_PATH} -> ${PROD_SRC}"
    fi
elif [[ -e "$LINK_PATH" ]]; then
    error "${LINK_PATH} exists and is not a symlink. Remove it manually."
    exit 1
else
    ln -s "$PROD_SRC" "$LINK_PATH"
    info "Created symlink: ${LINK_PATH} -> ${PROD_SRC}"
fi

# Ensure shiny user can traverse the symlink
chown -h shiny:shiny "$LINK_PATH" 2>/dev/null || true

# --- Step 2: Install missing + version-floored Python dependencies ---
# Presence-only check for packages we just need installed:
MISSING_PKGS=()
for pkg in pymoo SALib; do
    if ! "$SHINY_PIP" show "$pkg" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing missing packages: ${MISSING_PKGS[*]}"
    "$SHINY_PIP" install "${MISSING_PKGS[@]}" --quiet
fi

# Version-floored packages: a presence check cannot enforce a minimum, so upgrade
# unconditionally. cma <=3.3.0 breaks under numpy 2; shinyswatch <0.11 forces a sass
# compile under shiny 1.6.3; shinywidgets floor raised. shiny_deckgl must match prod's
# layer_legend_widget API (v1.9.2).
info "Ensuring version-floored packages (cma, shinyswatch, shinywidgets, shiny_deckgl)..."
"$SHINY_PIP" install --quiet --upgrade "cma>=4.0" "shinyswatch>=0.11" "shinywidgets>=0.7" "pyarrow>=14"
"$SHINY_PIP" install --quiet --upgrade "shiny_deckgl @ git+https://github.com/razinkele/shiny_deckgl.git@v1.9.2"

# Fail loudly if any floor is unmet after install.
"$SHINY_PIP" install --quiet "packaging" || true
"$SHINY_PYTHON" - <<'PYCHK'
import sys
from importlib.metadata import version
from packaging.version import Version
floors = {"cma": "4.0", "shinyswatch": "0.11", "shinywidgets": "0.7", "shiny": "1.6.3", "shiny_deckgl": "1.9.2"}
bad = []
for pkg, floor in floors.items():
    have = version(pkg)
    if Version(have) < Version(floor):
        bad.append(f"{pkg} {have} < {floor}")
if bad:
    print("DEPENDENCY FLOOR CHECK FAILED:", "; ".join(bad)); sys.exit(1)
print("dependency floors OK:", {k: version(k) for k in floors})
PYCHK
info "Python dependencies ensured."

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
# Calibration checkpoints must land in a service-user-writable dir. The source
# tree (WorkingDirectory) lives under another user's home and is read-only to
# 'shiny', so point OSMOSE_RESULTS_DIR at a systemd StateDirectory that systemd
# creates (owned by the service user) on every start.
StateDirectory=osmose/calibration_results
Environment=OSMOSE_RESULTS_DIR=/var/lib/osmose/calibration_results

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
# The app cold-starts in several seconds (numba/schema import), so a single
# immediate probe races startup and reports a spurious 000. Poll until 200 or
# the deadline, reporting the real state.
HTTP_CODE="000"
for _ in $(seq 1 20); do  # up to ~40s (2s interval)
    HTTP_CODE=$(curl -sS -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${APP_PORT}/" 2>/dev/null || echo "000")
    [[ "$HTTP_CODE" == "200" ]] && break
    sleep 2
done
if [[ "$HTTP_CODE" == "200" ]]; then
    info "HTTP check passed (port ${APP_PORT})."
else
    warn "HTTP check returned ${HTTP_CODE} after ~40s. Check: journalctl -u ${SERVICE_NAME} --no-pager -n 30"
fi

# --- Summary ---
echo ""
info "Deployment complete!"
echo "  Service:     ${SERVICE_NAME} (port ${APP_PORT})"
echo "  Prod source: ${PROD_SRC} @ ${DEPLOY_REF} (${DEPLOYED_SHA})"
echo "  Served via:  ${LINK_PATH} -> ${PROD_SRC}"
echo "  Deploy host: ${SOURCE_DIR} (ran this script; JAR source)"
echo "  Service:     ${SERVICE_FILE}"
echo ""
echo "  NOTE: Ensure nginx proxies /osmose/ to http://127.0.0.1:${APP_PORT}/"
echo "        (not to shiny-server on port 3838)"
echo ""
echo "  Commands:"
echo "    Restart:    sudo bash ${SOURCE_DIR}/deploy.sh --restart"
echo "    Logs:       journalctl -u ${SERVICE_NAME} -f"
echo "    Uninstall:  sudo bash ${SOURCE_DIR}/deploy.sh --uninstall"
