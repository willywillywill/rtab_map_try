FROM arm64v8/ros:noetic

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create non-root user and setup sudo
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    rm -rf /var/lib/apt/lists/*

# Install all system dependencies and ROS packages in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-opencv \
    nano \
    git \
    x11-apps \
    udev \
    v4l-utils \
    usbutils \
    mesa-utils \
    libuvc-dev \
    libusb-1.0-0-dev \
    ros-noetic-rviz \
    ros-noetic-turtlesim \
    ros-noetic-slam-gmapping \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-navigation \
    ros-noetic-hector-mapping \
    ros-noetic-rqt-tf-tree \
    ros-noetic-xacro \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-camera-info-manager \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-nodelet \
    ros-noetic-sensor-msgs \
    ros-noetic-image-geometry \
    ros-noetic-compressed-image-transport \
    ros-noetic-compressed-depth-image-transport \
    ros-noetic-rgbd-launch \
    ros-noetic-backward-ros \
    ros-noetic-joy \
    ros-noetic-teleop-twist-joy \
    ros-noetic-robot-state-publisher \
    ros-noetic-rtabmap-ros \
    ros-noetic-ackermann-steering-controller \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-ackermann-msgs \
    ros-noetic-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*


# 2. 在 Docker 內建立專屬的 Python 虛擬環境
RUN python3 -m venv /opt/yolo_env
# 3. 升級 pip 並安裝 Ultralytics 及 ROS Python 核心套件
RUN /opt/yolo_env/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/yolo_env/bin/pip install --no-cache-dir \
    ultralytics \
    rospkg \
    catkin_pkg \
    wheel \
    pyyaml \
    netifaces &&\
    /opt/yolo_env/bin/pip install --no-build-isolation --no-use-pep517 git+https://github.com/eric-wieser/ros_numpy.git

# Switch from root to user
USER $USERNAME

# Add user to video group
RUN sudo usermod --append --groups video $USERNAME

# Git configuration
RUN git config --global user.name willywillywill && \
    git config --global user.email 11013063@gm.hnvs.cy.edu.tw

# Rosdep update
RUN rosdep update

# Source setup scripts
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ./devel/setup.bash" >> ~/.bashrc
ENV PATH="/opt/yolo_env/bin:$PATH"
# Python packages
RUN pip install --no-cache-dir git+https://github.com/RobLibs/Rosmaster_Lib@V3.3.9 ipywidgets