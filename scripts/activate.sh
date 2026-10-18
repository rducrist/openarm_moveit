# Sourced by pixi on activation.
# Keep a caller-provided ROS_DOMAIN_ID (e.g. an isolated domain for mock tests); default 0.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Overlay the colcon workspace if it is built.
if [ -f "$PIXI_PROJECT_ROOT/install/local_setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$PIXI_PROJECT_ROOT/install/local_setup.bash"
fi

# Policy environment: tinydp (diffusion policy model code) is a source checkout, not installed.
if [ "${PIXI_ENVIRONMENT_NAME:-}" = "policy" ] && [ -d "$PIXI_PROJECT_ROOT/third_party/tinydp" ]; then
  export PYTHONPATH="$PIXI_PROJECT_ROOT/third_party/tinydp${PYTHONPATH:+:$PYTHONPATH}"
fi
