# Sourced by pixi on activation: overlay the colcon workspace if it is built.
if [ -f "$PIXI_PROJECT_ROOT/install/local_setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$PIXI_PROJECT_ROOT/install/local_setup.bash"
fi
