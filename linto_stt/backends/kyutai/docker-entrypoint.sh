#!/bin/bash
set -euo pipefail

echo "Starting Kyutai wrapper …"

###############################################################################
# 1 — Configure runtime user / group
###############################################################################
# Default to IDs that are unlikely to clash with system accounts.
USER_ID=${USER_ID:-33}
GROUP_ID=${GROUP_ID:-33}

# Default names (may be overridden below if the ID already exists).
USER_NAME=${USER_NAME:-appuser}
GROUP_NAME=${GROUP_NAME:-appgroup}

echo "⮑ Preparing user (UID=$USER_ID) and group (GID=$GROUP_ID)"

# --- group ---
if getent group "$GROUP_ID" >/dev/null 2>&1; then               # already exists
    GROUP_NAME=$(getent group "$GROUP_ID" | cut -d: -f1)
    echo "   • Re-using existing group: $GROUP_NAME"
else
    echo "   • Creating group: $GROUP_NAME"
    groupadd -g "$GROUP_ID" "$GROUP_NAME"
fi

# --- user ---
if id -u "$USER_ID" >/dev/null 2>&1; then                        # already exists
    USER_NAME=$(getent passwd "$USER_ID" | cut -d: -f1)
    echo "   • Re-using existing user: $USER_NAME"
else
    echo "   • Creating user: $USER_NAME"
    useradd -m -u "$USER_ID" -g "$GROUP_NAME" "$USER_NAME"
fi

# Ownership for application code (numeric IDs so the names may differ):
echo "⮑ Fixing ownership of /usr/src/app"
chown -R "${USER_ID}:${GROUP_ID}" /usr/src/app

###############################################################################
# 2 — Launch the requested service
###############################################################################
if [ -z "${SERVICE_MODE:-}" ]; then
    echo "ERROR • SERVICE_MODE not defined (must be one of: http | websocket)"
    exit 1
fi

echo "⮑ Launching in SERVICE_MODE=$SERVICE_MODE"

case "$SERVICE_MODE" in
    websocket)
        exec gosu "${USER_ID}:${GROUP_ID}" \
            python3 websocket/websocketserver.py
        ;;
    http)
        exec gosu "${USER_ID}:${GROUP_ID}" \
            python3 http_server/ingress.py --debug
        ;;
    *)
        echo "ERROR • Unknown SERVICE_MODE '$SERVICE_MODE' (expected http | websocket)"
        exit 1
        ;;
esac