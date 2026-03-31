#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -xe

echo "=== Ollama Heavyweight Auto-Provisioner (Vast.ai) ==="

# Configuration
VAST_TEMPLATE="${VAST_TEMPLATE:-c166c11f035d3a97871a23bd32ca6aba}"
VAST_CMD="./vast.py"

# 1. Ensure required tools are available
if [ ! -f "$VAST_CMD" ] && ! command -v vast &> /dev/null; then
    echo "Error: vast CLI not found. Ensure ./vast.py exists or 'vast' is in PATH."
    exit 1
fi
# Use vast directly if it's in PATH, otherwise use ./vast.py
if command -v vast &> /dev/null; then
    VAST_CMD="vast"
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq not found. Ensure jq is installed."
    exit 1
fi

echo "Searching for the best available Vast.ai instance..."

# Search for the best instance matching reliability and DPH criteria
# Grabbing JSON output via --raw and parsing first result
SEARCH_JSON=$($VAST_CMD search offers 'reliability > 0.99 verified=True dph>=2.3 dph<=3' -o 'gpu_total_ram-,dlperf-,total_flops-' -d --raw 2>/dev/null)

if [ -z "$SEARCH_JSON" ] || [ "$SEARCH_JSON" == "null" ] || [ "$SEARCH_JSON" == "[]" ]; then
    echo "Error: No instances found matching the criteria."
    exit 1
fi

INSTANCE_ID=$(echo "$SEARCH_JSON" | jq -r '.[0].id')
TOTAL_VRAM_MB=$(echo "$SEARCH_JSON" | jq -r '.[0].gpu_total_ram')

echo "Top Instance ID selected: $INSTANCE_ID"
echo "Total System VRAM detected on instance: ${TOTAL_VRAM_MB} MB"

# 2. Curated list of top-tier tool-capable models (Heavyweights First)
# Format: "model_name required_vram_in_mb"
# Ordered from highest capability/size to lowest. 
# VRAM requirements are heavily padded to allow for maximum context windows.
MODELS=(
    "llama3.1:405b 260000"       # ~231GB for weights + ~30GB context buffer
    "mixtral:8x22b 95000"        # ~80GB for weights + ~15GB context buffer
    "command-r-plus:104b 75000"  # ~60GB for weights + ~15GB context buffer
    "qwen2.5-coder:72b 55000"    # ~48GB for weights + ~7GB context buffer
    "llama3.1:70b 50000"         # ~40GB for weights + ~10GB context buffer
    "mixtral:8x7b 32000"         # Fallback
    "command-r:35b 28000"        # Fallback
    "llama3.1:8b 8000"           # Deep fallback
)

SELECTED_MODEL=""
REQUIRED_VRAM=0

# 3. Find the largest model that fits the available instance VRAM
for entry in "${MODELS[@]}"; do
    read -r MODEL_NAME REQ_VRAM <<< "$entry"
    
    # vast float MB cleanup to interger
    INT_VRAM=${TOTAL_VRAM_MB%.*}
    if (( INT_VRAM >= REQ_VRAM )); then
        SELECTED_MODEL=$MODEL_NAME
        REQUIRED_VRAM=$REQ_VRAM
        echo "Selected Model: $SELECTED_MODEL (Requires ~${REQUIRED_VRAM} MB for weights + context)"
        break
    fi
done

if [ -z "$SELECTED_MODEL" ]; then
    echo "Error: Not enough VRAM to run any of the configured models on this instance."
    exit 1
fi

# 4. Calculate required disk space in GB (Total VRAM + 15%)
# Convert required VRAM (MB) to GB, then multiply by 1.15
# Added a base 32GB for OS filesystem and Docker images padding to prevent disk pressure
REQUIRED_DISK_GB=$(awk "BEGIN {print int(($REQUIRED_VRAM / 1024) * 1.15) + 32}")
echo "Calculated required disk space: ${REQUIRED_DISK_GB} GB (Model size + 15% + base OS buffer)"

# 5. Extract template environment options & Create Vast.ai instance
echo "Provisioning instance $INSTANCE_ID on Vast.ai..."

echo "Fetching template details to extract env options..."
# Changed to `show` as vast CLI uses show templates to get the full list
TEMPLATE_JSON=$($VAST_CMD show templates --raw 2>/dev/null)

# Extract the env value correctly without blind fallbacks (check hash, id, and hash_id)
TEMPLATE_ENV=$(echo "$TEMPLATE_JSON" | jq -r ".[] | select(.id == \"$VAST_TEMPLATE\" or .hash == \"$VAST_TEMPLATE\" or .hash_id == \"$VAST_TEMPLATE\") | .env" | head -n 1)

if [ "$TEMPLATE_ENV" == "null" ] || [ -z "$TEMPLATE_ENV" ]; then
    echo "Warning: Template $VAST_TEMPLATE not found or empty. Creating without extra env."
    TEMPLATE_ENV=""
fi

# Append the volume requirement to the dynamically extracted env options
FINAL_ENV="$TEMPLATE_ENV -v /workspace"

echo "Template Target: $VAST_TEMPLATE"
echo "Extracted Env: $FINAL_ENV"

echo ""
read -r -p "Do you want to proceed with creating this instance? (default: no) [yes/N]: " USER_CONFIRM
if [[ "$USER_CONFIRM" != "yes" ]]; then
    echo "Instance creation aborted by user (must type 'yes' to proceed)."
    exit 0
fi

# We provision the instance with the calculated --disk size and the dynamically extracted env
CREATE_OUTPUT=$($VAST_CMD create instance $INSTANCE_ID \
    --image "$VAST_TEMPLATE" \
    --disk $REQUIRED_DISK_GB \
    --env "$FINAL_ENV" \
    --raw)

# Parse out the new instance ID
NEW_CONTRACT=$(echo "$CREATE_OUTPUT" | jq -r '.new_contract // .id // empty')

if [ -z "$NEW_CONTRACT" ]; then
    echo "Failed to create instance or parse contract ID. Output:"
    echo "$CREATE_OUTPUT"
    exit 1
fi

echo "=== Provisioning Complete! Contract: $NEW_CONTRACT ==="
echo "Model targeted for setup on startup: $SELECTED_MODEL"
echo "Waiting for instance to be fully ready to retrieve external URLs (this may take a few minutes)..."

# Poll until state becomes running to get the URLs
for i in {1..30}; do
    INSTANCE_INFO=$($VAST_CMD show instances --raw 2>/dev/null | jq -r ".[] | select(.id == $NEW_CONTRACT)")
    STATUS=$(echo "$INSTANCE_INFO" | jq -r '.actual_status // .state // "unknown"')
    if [ "$STATUS" == "running" ]; then
        break
    fi
    sleep 10
done

SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.ssh_host')
SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ssh_port')
MACHINE_ID=$(echo "$INSTANCE_INFO" | jq -r '.machine_id')

echo ""
echo "=== Preparing Model via Vast CLI ==="
echo "Pulling $SELECTED_MODEL directly on the instance..."
echo "(WARNING: Heavyweight models take significant time to download.)"

# Use the native vast cli execution tool instead of raw SSH polling
$VAST_CMD execute "$NEW_CONTRACT" "ollama pull '$SELECTED_MODEL' && echo 'Success! Model pulled ready.' || echo 'Failed to pull model.'"

echo ""
echo "=== Instance External URLs & Connection Details ==="
echo "Final Status: $STATUS"

echo "SSH Access: ssh -p $SSH_PORT root@$SSH_HOST"
echo "Service URLs:"

# Extract and output format for both standard and vast.ai HTTPS proxy URLs
echo "$INSTANCE_INFO" | jq -r '
    if .ports then
        .ports | to_entries | .[] |
        (if .value[0].HostIp then
            " -> Internal Port " + .key + "\n" +
            "    Standard IP URL: http://" + .value[0].HostIp + ":" + .value[0].HostPort + "\n" +
            "    Vast HTTPS Proxy: https://server-'"$MACHINE_ID"'.vast.ai:" + .value[0].HostPort
        else
            " -> Internal Port " + .key + " (Not publicly mapped yet)"
        end)
    else
        " -> No port mappings found."
    end
'

echo ""
echo "Monitor instances using: $VAST_CMD show instances"