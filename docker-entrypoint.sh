#!/bin/bash
set -euo pipefail
set -a


if [ -z "${SERVICE_NAME:-}" ]; then
    echo "ERROR: SERVICE_NAME not set (e.g. kaldi, whisper, nemo, kyutai)"
    exit 1
fi

echo "Starting ${SERVICE_NAME} service…"

###############################################################################
# 1 — Runtime user / group
###############################################################################
USER_ID=${USER_ID:-33}
GROUP_ID=${GROUP_ID:-33}
USER_NAME="appuser"
GROUP_NAME="appgroup"
PORT=${PORT:-80}
IP=${IP:-0.0.0.0}

echo "=> Preparing user (UID=$USER_ID, GID=$GROUP_ID)"

# Check if a group with the specified GID already exists
if getent group "$GROUP_ID" >/dev/null 2>&1; then
    GROUP_NAME=$(getent group "$GROUP_ID" | cut -d: -f1)
else
    groupadd -g "$GROUP_ID" "$GROUP_NAME"
fi

# Check if a user with the specified UID already exists
if id -u "$USER_ID" >/dev/null 2>&1; then
    USER_NAME=$(getent passwd "$USER_ID" | cut -d: -f1)
else
    useradd -m -u "$USER_ID" -g "$GROUP_NAME" "$USER_NAME"
fi

# App directory ownership
chown -R "${USER_ID}:${GROUP_ID}" /usr/src/app

# Home directory setup (needed for model downloads)
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
if [ ! -d "$USER_HOME" ]; then
    mkdir -p "$USER_HOME"
fi
chown -R "$USER_NAME:$GROUP_NAME" "$USER_HOME"
chmod -R u+rwx "$USER_HOME"

# /opt permissions (needed for model downloads)
chmod g+rwx /opt 2>/dev/null || true
usermod -aG "$(stat -c %G /opt)" "$USER_NAME" 2>/dev/null || true

###############################################################################
# 2 — Launch service
###############################################################################
if [ -z "${SERVICE_MODE:-}" ]; then
    echo "ERROR: SERVICE_MODE not set (expected: http | task | websocket)"
    exit 1
fi

echo "=> SERVICE_MODE=$SERVICE_MODE"

case "$SERVICE_MODE" in
    http)
        echo "Launching HTTP server"
        exec gosu "$USER_NAME" python -m linto_stt -m http -b "$SERVICE_NAME" -p "$PORT" -i "$IP"
        ;;

    websocket)
        echo "Launching websocket server"
        exec gosu "$USER_NAME" python -m linto_stt -m websocket -b "$SERVICE_NAME" -p "$PORT" -i "$IP"
        ;;

    task)
        if [ -z "${SERVICES_BROKER:-}" ]; then
            echo "ERROR: SERVICES_BROKER not set, cannot start celery worker"
            exit 1
        fi

        BROKER_HOST=$(echo "$SERVICES_BROKER" | cut -d'/' -f 3)
        ./wait-for-it.sh "$BROKER_HOST" --timeout=20 --strict -- \
            echo "$SERVICES_BROKER (Service Broker) is up" || exit 1

        echo "Launching celery worker"
        exec gosu "$USER_NAME" python -m linto_stt -m task -b "$SERVICE_NAME"
        ;;
    *)
        echo "ERROR: Unknown SERVICE_MODE '$SERVICE_MODE' (expected: http | task | websocket)"
        exit 1
        ;;
esac