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
#         sudo OSMOSE_DEPLOY_CSHARP=1 bash deploy.sh   # also build + self-test the C# engine

set -euo pipefail

APP_NAME="osmose"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
SHINY_ROOT="/srv/shiny-server"
SHINY_PYTHON="/opt/micromamba/envs/shiny/bin/python3"
SHINY_PIP="/opt/micromamba/envs/shiny/bin/pip"
# pip runs AS SHINY (see as_shiny below): the env is shiny-owned. --root-user-action=ignore is
# inert for a non-root user and kept so a root invocation would not warn. Use
# 'as_shiny "${PIP_INSTALL[@]}" <pkgs>' for installs. --no-user: pip falls back to a user-site
# install when site-packages is not writable, and the floor check would then pass on a copy in
# shiny's home; with it, an install pip cannot write into the env fails the deploy instead.
PIP_INSTALL=("$SHINY_PIP" install --root-user-action=ignore --no-user)
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

# The C# engine (docs/superpowers/specs/2026-11-17-csharp-engine-design.md, section 9.7) is
# OPT-IN: the web app does not offer it before the Phase 2 UI work, so a plain deploy installs
# nothing for it. OSMOSE_DEPLOY_CSHARP=1 installs the .NET 10 SDK into DOTNET_DIR if absent,
# builds the assembly inside the prod clone and self-tests it as the service user (Step 6).
DEPLOY_CSHARP="${OSMOSE_DEPLOY_CSHARP:-0}"
DOTNET_DIR="/opt/dotnet"

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

# Deploying FROM the prod clone checks out the new ref and then dies on `cp` copying the JARs
# onto themselves (set -e) -- before the restart, leaving new code on disk under the old
# process (2026-10-29). --restart/--uninstall from the clone are fine; a deploy is not.
if [[ "$SOURCE_DIR" -ef "$PROD_SRC" ]]; then
    error "Run the deploy from the dev checkout, not the prod clone (${PROD_SRC})."
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

# --- Root never executes, or writes through, anything the service user can modify ---
# The prod clone (.git included) and the shared env (/opt/micromamba/envs/shiny, its python3,
# pip and site-packages) are owned by shiny. Run as root, `git checkout` would execute the
# clone's hooks, core.fsmonitor and filter drivers; the env's python3 would execute its
# sitecustomize and .pth files (or a replaced interpreter); and `cp` into the clone would follow
# a planted symlink. A compromised service account would then get root at the next deploy. So
# every such step runs as shiny: it gains nothing that user did not already have. From /, so a
# working directory shiny cannot enter does not break git or pip. Under setsid, so the child
# has NO controlling terminal: a shiny process sharing root's terminal could otherwise push
# keystrokes into it (the TIOCSTI ioctl) for root's shell to run after the deploy. This kernel
# disables that (dev.tty.legacy_tiocsti=0, 2026-11-18), but a sysctl is not a guarantee.
SHINY_HOME="$(getent passwd shiny | cut -d: -f6)"
as_shiny() { (cd / && setsid -w runuser -u shiny -- env HOME="$SHINY_HOME" "$@"); }

# --- Step 0: Install missing + version-floored Python dependencies ---
# BEFORE the checkout: pip runs as shiny (above), and a failure here (a network error, or a
# package below its floor that shiny cannot replace because it is root-owned; DEPLOY.md, "The
# shared environment") then stops the deploy with the previous ref still checked out, rather
# than leaving new code on disk under the old process. Installs that completed before the
# failure stay in the shared env; they only raise packages to their floors, which the running
# code already satisfies.
# Clean up leftover "~"-prefixed dist-info dirs (broken partial installs from interrupted
# pip upgrades, e.g. ~inycss2 / ~hinyswatch / ~ma). pip warns "Ignoring invalid distribution"
# for each on every command; removing them is safe (the real packages are the unprefixed dirs).
SHINY_SITE_PACKAGES="$(as_shiny "$SHINY_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -d "$SHINY_SITE_PACKAGES" ]]; then
    LEFTOVER=$(find "$SHINY_SITE_PACKAGES" -maxdepth 1 -name '~*' 2>/dev/null)
    if [[ -n "$LEFTOVER" ]]; then
        info "Removing leftover partial-install dirs in $SHINY_SITE_PACKAGES"
        # Cosmetic (pip only warns about them), so a leftover shiny cannot remove, e.g. one an
        # older root deploy left root-owned, is reported, not fatal.
        as_shiny find "$SHINY_SITE_PACKAGES" -maxdepth 1 -name '~*' -exec rm -rf {} + \
            || warn "Could not remove all of: ${LEFTOVER} (root-owned? DEPLOY.md, \"The shared environment\")"
    fi
fi

# Presence-only check for packages we just need installed:
MISSING_PKGS=()
for pkg in pymoo SALib; do
    if ! as_shiny "$SHINY_PIP" show "$pkg" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing missing packages: ${MISSING_PKGS[*]}"
    as_shiny "${PIP_INSTALL[@]}" "${MISSING_PKGS[@]}" --quiet
fi

# Version floors. pip changes a package only when it is missing or BELOW its floor, never to
# reach a newer release: the env is shared with other apps and conda-managed (DEPLOY.md, "The
# shared environment"), so replacing a package that already meets its floor can break another
# app or stop the deploy on a root-owned file. A specifier without --upgrade is exactly that:
# "already satisfied" is a no-op, and a missing or older package gets the newest release that
# meets the floor. (Until 2026-11-21 this line passed --upgrade, which would have replaced cma
# 4.0.0 with 4.6.0.) cma <=3.3.0 breaks under numpy 2; shinyswatch <0.11 forces a sass compile
# under shiny 1.6.3; shinywidgets floor raised. packaging is needed by the guards below.
info "Ensuring version-floored packages (cma, shinyswatch, shinywidgets, shiny_deckgl, pyvis)..."
as_shiny "${PIP_INSTALL[@]}" --quiet "packaging" "cma>=4.0" "shinyswatch>=0.11" "shinywidgets>=0.7" "pyarrow>=14"
# shiny_deckgl comes from a git tag: v1.9.2 is the release osmose was built against (its
# layer_legend_widget API); v1.11.0 was checked on 1.16.1 too (2026-11-21). Install it only when
# missing or below 1.9.2, since pip reinstalls a git-URL requirement under --upgrade every time.
# This test used to be "not exactly 1.9.2", which would have downgraded the 1.16.1 that the
# server's other apps (AQUABC, CENOP) need.
if ! as_shiny "$SHINY_PYTHON" -c 'import sys; from importlib.metadata import version; from packaging.version import Version; sys.exit(Version(version("shiny_deckgl")) < Version("1.9.2"))' 2>/dev/null; then
    as_shiny "${PIP_INSTALL[@]}" --quiet --upgrade "shiny_deckgl @ git+https://github.com/razinkele/shiny_deckgl.git@v1.9.2"
fi
# pyvis is a FORK (razinkele/pyvis "Optimized Edition", versioned 4.x); PyPI's `pyvis` is a
# different lineage that stops at 0.3.2. The fork ships under TWO distribution names: `pyvis`
# on the maintainer's conda channel, and `pyvis-optimized` on PyPI -- both install the same
# `pyvis` import package. The shared env is used by other apps and may already carry either
# (2026-10-29: `pyvis-optimized 4.5.0`, and NO `pyvis` dist at all, so the old
# `version("pyvis")` check aborted the deploy). So: key everything on the IMPORT package's own
# `__version__`, not a dist name, and only install `pyvis-optimized` when that is missing or
# below the floor -- never overwrite a conda-managed fork that already satisfies it. A presence
# check (`pip show`, `import pyvis`) would pass on upstream 0.3.2; the floor is what catches it.
if ! as_shiny "$SHINY_PYTHON" -c 'import sys, pyvis; from packaging.version import Version; sys.exit(Version(getattr(pyvis, "__version__", "0")) < Version("4.2"))' 2>/dev/null; then
    info "pyvis fork missing or below 4.2; installing pyvis-optimized from PyPI..."
    as_shiny "${PIP_INSTALL[@]}" --quiet --upgrade "pyvis-optimized>=4.2"
fi

# Fail loudly if any floor is unmet after install.
as_shiny "$SHINY_PYTHON" - <<'PYCHK'
import sys
from importlib.metadata import version
from packaging.version import Version
floors = {"cma": "4.0", "shinyswatch": "0.11", "shinywidgets": "0.7", "shiny": "1.6.3",
          "shiny_deckgl": "1.9.2"}
have = {k: version(k) for k in floors}
# pyvis by import-package version: its dist name is `pyvis` or `pyvis-optimized` (see above).
import pyvis
floors["pyvis"] = "4.2"
have["pyvis"] = getattr(pyvis, "__version__", "0")
bad = [f"{pkg} {have[pkg]} < {floor}" for pkg, floor in floors.items()
       if Version(have[pkg]) < Version(floor)]
if bad:
    print("DEPENDENCY FLOOR CHECK FAILED:", "; ".join(bad)); sys.exit(1)
print("dependency floors OK:", have)
PYCHK
info "Python dependencies ensured."

# --- Step 1: Sync the prod source clone ---
# Clone once, then fetch + checkout the target ref on every deploy. This is the
# ONLY moment prod's source changes (followed by a restart in Step 4), so the
# running process never reads a file that is being edited underneath it.
# Older deploys let ROOT operate on this shiny-owned repository (safe.directory overrides git's
# guard against exactly that, CVE-2022-24765). Git now runs as shiny, so withdraw the grant.
git config --global --fixed-value --unset-all safe.directory "$PROD_SRC" 2>/dev/null || true
if [[ -d "${PROD_SRC}/.git" ]]; then
    info "Updating prod clone at ${PROD_SRC} (fetch ${DEPLOY_REF})..."
    # Root never writes into the clone, not even to re-own it: a hard link shiny planted there
    # would carry the change to a file outside it. Anything an older, root-git deploy left
    # root-owned would make git as shiny fail half-way, so refuse first (read-only check);
    # DEPLOY.md, "Root-owned files in the clone", has the fix.
    ROOT_OWNED="$(find "$PROD_SRC" -xdev -user root -print -quit 2>/dev/null || true)"
    if [[ -n "$ROOT_OWNED" ]]; then
        error "Root-owned files in ${PROD_SRC} (first: ${ROOT_OWNED}). See DEPLOY.md, \"Root-owned files in the clone\"."
        exit 1
    fi
    as_shiny git -C "$PROD_SRC" fetch --quiet --prune origin
elif [[ -e "$PROD_SRC" ]]; then
    error "${PROD_SRC} exists but is not a git clone. Remove it manually."
    exit 1
else
    info "Creating prod clone: ${REPO_URL} -> ${PROD_SRC}"
    install -d -o shiny -g shiny "$PROD_SRC"  # root-owned parent: root creates the empty dir
    if ! as_shiny git clone --quiet "$REPO_URL" "$PROD_SRC"; then
        rmdir "$PROD_SRC" 2>/dev/null || true  # the dir made above, only if git left it empty
        error "git clone of ${REPO_URL} failed; nothing was deployed. Re-run once it is reachable."
        exit 1
    fi
fi
# The osmose-java JARs are NOT tracked in git (so they are absent from a fresh clone). Provision
# them from the deploy host's tree so Java-engine runs remain available. Best-effort: a failure is
# a warning (prod's primary path is the Python engine; a missing JAR only disables Java runs). It
# runs before the checkout, and each JAR lands under a temporary name and is then renamed, so a
# failed copy never leaves a truncated JAR. Root reads the JARs (the deploy host's tree may be
# closed to shiny) into a fresh root-owned directory shiny can read; shiny copies them in.
provision_jars() {
    as_shiny mkdir -p "${PROD_SRC}/osmose-java" || return 1
    local jar name
    for jar in "$JAR_STAGE"/*.jar; do
        name="$(basename "$jar")"
        as_shiny cp -f "$jar" "${PROD_SRC}/osmose-java/.${name}.part" || return 1
        as_shiny mv -f "${PROD_SRC}/osmose-java/.${name}.part" "${PROD_SRC}/osmose-java/${name}" \
            || return 1
    done
}
if compgen -G "${SOURCE_DIR}/osmose-java/"*.jar >/dev/null 2>&1; then
    JAR_STAGE="$(mktemp -d)"
    chmod 755 "$JAR_STAGE"
    if install -m 644 "${SOURCE_DIR}/osmose-java/"*.jar "$JAR_STAGE/" && provision_jars; then
        info "Provisioned Java JAR(s) into ${PROD_SRC}/osmose-java/."
    else
        as_shiny find "${PROD_SRC}/osmose-java" -maxdepth 1 -name '.*.part' -delete 2>/dev/null || true
        warn "Could not provision the Java JAR(s); Java-engine runs may be unavailable in prod."
    fi
    rm -rf "$JAR_STAGE"
else
    warn "No osmose-java/*.jar in ${SOURCE_DIR}; Java-engine runs will be unavailable in prod."
fi

# The C# assembly is gitignored, so `checkout --force` keeps the previous ref's build, and
# csharp_engine_available() only compares the feature catalogue: a stale assembly would load
# against the new Python code. Remove it on EVERY deploy, BEFORE the checkout, so a failed
# removal stops the deploy with the previous ref and its build still consistent; Step 6
# rebuilds it from the new ref when OSMOSE_DEPLOY_CSHARP=1. As shiny, not root: the clone is
# shiny-writable, and root's rm -rf would follow a symlink planted above bin/ out of the clone.
if [[ -e "${PROD_SRC}/osmose/engine_cs/bin" || -L "${PROD_SRC}/osmose/engine_cs/bin" ]]; then
    as_shiny rm -rf "${PROD_SRC}/osmose/engine_cs/bin"
fi
as_shiny git -C "$PROD_SRC" checkout --quiet --force --detach "$DEPLOY_REF"
DEPLOYED_SHA="$(as_shiny git -C "$PROD_SRC" rev-parse --short HEAD)"
info "Prod source at ${DEPLOY_REF} (${DEPLOYED_SHA})."


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
# Calibration checkpoints land in a systemd StateDirectory that systemd creates,
# owned by the service user, on every start.
# NOTE: this comment used to say the source tree "lives under another user's home
# and is read-only to 'shiny'". That was measured on 2026-09-14 and is FALSE --
# the tree is owned by and writable to the service user, and this unit sets no
# ProtectSystem/ReadOnlyPaths. Do not reason from the old claim: it was copied
# into DEPLOY.md and from there into a wrong prediction about the feedback store
# that took a review round to retract. OSMOSE_RESULTS_DIR is still correct and
# should stay -- keeping state out of the source tree survives a redeploy.
StateDirectory=osmose/calibration_results
Environment=OSMOSE_RESULTS_DIR=/var/lib/osmose/calibration_results
# The C# engine's runtime (Step 6, opt-in). Inert when the directory is absent: the engine is
# then simply unavailable and the app does not offer it.
Environment=DOTNET_ROOT=${DOTNET_DIR}

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

# --- Step 6 (opt-in): the C# engine ---
# Runs AFTER the restart on purpose. Nothing loads the CLR at startup and availability is checked
# per call, so the service is already serving the new ref without the engine; a failure here only
# means "no C# engine", reported loudly with exit 1 -- never new code on disk under the old
# process (the 2026-10-29 failure mode of an abort between checkout and restart).
CSHARP_RESULT="not built (OSMOSE_DEPLOY_CSHARP=1 builds it)"
if [[ "$DEPLOY_CSHARP" == "1" ]]; then
    csharp_fail() {
        error "C# engine: $*"
        error "The app is running ${DEPLOYED_SHA} WITHOUT the C# engine."
        exit 1
    }
    if ! "${DOTNET_DIR}/dotnet" --list-sdks 2>/dev/null | grep -q '^10\.'; then
        info "Installing the .NET 10 SDK into ${DOTNET_DIR} (dotnet-install.sh)..."
        DOTNET_INSTALLER="$(mktemp)"
        curl -fsSL https://dot.net/v1/dotnet-install.sh -o "$DOTNET_INSTALLER" \
            || csharp_fail "could not download dotnet-install.sh"
        bash "$DOTNET_INSTALLER" --channel 10.0 --install-dir "$DOTNET_DIR" \
            || csharp_fail "the .NET SDK install failed"
        rm -f "$DOTNET_INSTALLER"
    fi
    as_shiny "${PIP_INSTALL[@]}" --quiet "pythonnet>=3.2.1" || csharp_fail "pip install pythonnet failed"
    info "Building the C# engine (as shiny, into a staging directory)..."
    # AS THE SERVICE USER, never as root: the clone is shiny-writable (Step 1) and MSBuild project
    # files are code -- an untracked Directory.Build.targets above the project, or a
    # *.nuget.g.targets in obj/, would otherwise run as root. Built as shiny it gains nothing that
    # user did not already have. Leftover build state is cleared first (also as shiny: git runs
    # hooks and core.fsmonitor from the shiny-owned .git/config), and the SDK's first-run and
    # NuGet files go to a throwaway home, since shiny's own home may not exist.
    # The build is published to a STAGING directory and self-tested there. Only a build that
    # passes is moved into osmose/engine_cs/bin, so a failed build never becomes loadable.
    as_shiny git -C "$PROD_SRC" clean -fdxq -- csharp/ \
        || csharp_fail "could not clear old build state in ${PROD_SRC}/csharp"
    BUILD_HOME="$(mktemp -d)"
    chown shiny:shiny "$BUILD_HOME"
    STAGE="${BUILD_HOME}/engine"
    LIVE_BIN="${PROD_SRC}/osmose/engine_cs/bin"
    drop_build_home() { as_shiny rm -rf "$BUILD_HOME" || true; }
    # DOTNET_ROOT explicitly: the build script would otherwise fall back to ~/.dotnet.
    # OSMOSE_CSHARP_ENGINE_DIR, explicitly: where the build publishes and the self-test loads from.
    as_shiny env HOME="$BUILD_HOME" DOTNET_CLI_HOME="$BUILD_HOME" \
        NUGET_PACKAGES="${BUILD_HOME}/nuget" DOTNET_ROOT="$DOTNET_DIR" \
        OSMOSE_CSHARP_ENGINE_DIR="$STAGE" \
        bash "${PROD_SRC}/scripts/build_csharp_engine.sh" -v quiet \
        || { drop_build_home; csharp_fail "the build failed"; }
    # As the SERVICE runs it: its user, interpreter, working copy and DOTNET_ROOT. Run as root it
    # could find another .NET (resolve_dotnet_root falls back to ~/.dotnet) and pass falsely.
    as_service() {
        (cd "$LINK_PATH" && setsid -w runuser -u shiny -- env HOME="$SHINY_HOME" DOTNET_ROOT="$DOTNET_DIR" "$@")
    }
    if ! as_service OSMOSE_CSHARP_ENGINE_DIR="$STAGE" "$SHINY_PYTHON" -m osmose.engine_cs.selftest; then
        drop_build_home
        csharp_fail "the self-test failed (its output is above); the build was not activated"
    fi
    as_shiny rm -rf "$LIVE_BIN" && as_shiny mv "$STAGE" "$LIVE_BIN" \
        || { drop_build_home; csharp_fail "could not move the tested build into ${LIVE_BIN}"; }
    drop_build_home
    # The self-test loaded the staged copy; confirm the service's own path (no override) sees it.
    as_service "$SHINY_PYTHON" -c "from osmose.engine_cs import csharp_engine_available as a
ok, why = a(); raise SystemExit(0 if ok else why)" \
        || { as_shiny rm -rf "$LIVE_BIN"; csharp_fail "the activated build is not loadable"; }
    CSHARP_RESULT="built from ${DEPLOYED_SHA}, self-tested, activated"
fi

# --- Summary ---
echo ""
info "Deployment complete!"
echo "  Service:     ${SERVICE_NAME} (port ${APP_PORT})"
echo "  Prod source: ${PROD_SRC} @ ${DEPLOY_REF} (${DEPLOYED_SHA})"
echo "  Served via:  ${LINK_PATH} -> ${PROD_SRC}"
echo "  Deploy host: ${SOURCE_DIR} (ran this script; JAR source)"
echo "  C# engine:   ${CSHARP_RESULT}"
echo "  Service:     ${SERVICE_FILE}"
echo ""
echo "  NOTE: Ensure nginx proxies /osmose/ to http://127.0.0.1:${APP_PORT}/"
echo "        (not to shiny-server on port 3838)"
echo ""
echo "  Commands:"
echo "    Restart:    sudo bash ${SOURCE_DIR}/deploy.sh --restart"
echo "    Logs:       journalctl -u ${SERVICE_NAME} -f"
echo "    Uninstall:  sudo bash ${SOURCE_DIR}/deploy.sh --uninstall"
