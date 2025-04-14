#!/usr/bin/env bash

set -e

# Verify required environment variables
if [ -z "$GATEWAY_URL" ] || [ -z "$SERVICE_TYPE" ] || [ -z "$SERVICE_ENDPOINT"] || [ -z "$SERVICE_PORT"] || [ -z "$SERVICE_NAME"] || [ -z "$SERVICE_HOST"] || [ -z "$SERVICE_HEALTHCHECK"]; then
    echo "Error: GATEWAY_URL, SERVICE_TYPE, SERVICE_ENDPOINT, SERVICE_PORT, SERVICE_NAME, SERVICE_HOST, SERVICE_HEALTHCHECK must be set." >&2
    exit 1
fi

TEMPLATE_FILE="heartbeat.json"

# Check if the template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: $TEMPLATE_FILE not found." >&2
    exit 1
fi

json_payload=$(envsubst < "$TEMPLATE_FILE")
curl -X POST -H "Content-Type: application/json" -d "$json_payload" "${GATEWAY_URL}?type=${SERVICE_TYPE}"
