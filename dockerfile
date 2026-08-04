FROM arm64v8/ros:noetic
#FROM osrf/ros:noetic-desktop-full
# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME


RUN apt install python3-pip -y
RUN apt-get install python3-opencv -y
RUN apt install -y ros-noetic-rviz
RUN apt install -y nano
RUN apt install ros-noetic-turtlesim -y
RUN apt install ros-noetic-slam-gmapping -y
RUN apt install ros-noetic-teleop-twist-keyboard
RUN apt install ros-noetic-navigation -y
RUN apt install ros-noetic-hector-mapping
RUN apt install ros-noetic-rqt-tf-tree -y
RUN apt install -y ros-noetic-xacro
RUN apt install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-camera-info-manager \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-nodelet \
    ros-noetic-sensor-msgs \
    ros-noetic-image-geometry \
    ros-noetic-compressed-image-transport \
    ros-noetic-compressed-depth-image-transport \
    libuvc-dev \
    libusb-1.0-0-dev \
    ros-noetic-rgbd-launch \
    ros-noetic-backward-ros \
    udev \
    v4l-utils \
    ros-noetic-joy \
    ros-noetic-teleop-twist-joy\
    usbutils \
    ros-noetic-robot-state-publisher \
    ros-noetic-rtabmap-ros \
    mesa-utils

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git
RUN git config --global user.name willywillywill && \
    git config --global user.email 11013063@gm.hnvs.cy.edu.tw

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source ./devel/setup.bash" >> ~/.bashrc
# x11 apps for testing GUI applications
RUN sudo apt install -y x11-apps

RUN pip install git+https://github.com/RobLibs/Rosmaster_Lib@V3.3.9
RUN pip install ipywidgets

################################
## ADD ANY CUSTOM SETUP BELOW ##
################################



