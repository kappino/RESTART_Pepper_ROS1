#!/bin/bash

echo "🔧 Setting up Conda environment with ROS integration"

# 1. Create Conda environment if not exists
ENV_NAME="restart"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.8 -y
fi

# 2. Activate environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 3. Install essential ROS Python packages in Conda
conda install -c conda-forge empy catkin_pkg rospkg rosdep -y
pip install pyyaml

# 4. Initialize rosdep (if not done before)
sudo rosdep init
rosdep update

# 5. Setup activation/deactivation hooks
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

# 6. ROS activation hook
cat > "$ACTIVATE_DIR/ros_activate.sh" << 'EOF'
#!/bin/bash
# Reset environment variables
unset PYTHONPATH

# Source ROS
source /opt/ros/noetic/setup.bash

# Source catkin workspace if exists
if [ -f "$(pwd)/catkin_ws/devel/setup.bash" ]; then
    source "$(pwd)/catkin_ws/devel/setup.bash"
elif [ -f "$(pwd)/devel/setup.bash" ]; then
    source "$(pwd)/devel/setup.bash"
fi

# Add ROS Python packages to PYTHONPATH
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH

# Ensure CMake uses Conda Python
export PYTHON_EXECUTABLE=$(which python)
EOF

# 7. ROS deactivation hook
cat > "$DEACTIVATE_DIR/ros_deactivate.sh" << 'EOF'
#!/bin/bash
unset ROS_PACKAGE_PATH
unset ROS_ROOT
unset ROS_MASTER_URI
unset ROS_VERSION
unset ROS_LOCALHOST_ONLY
unset ROS_DISTRO
unset PYTHON_EXECUTABLE
EOF

# 8. Make hooks executable
chmod +x "$ACTIVATE_DIR/ros_activate.sh" "$DEACTIVATE_DIR/ros_deactivate.sh"

echo "✅ Setup completed. Environment '$ENV_NAME' is now ROS-ready."
echo "To activate: conda activate $ENV_NAME"
echo "To build: cd catkin_ws && catkin_make -DPYTHON_EXECUTABLE=$(which python)"