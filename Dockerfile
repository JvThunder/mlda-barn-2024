FROM ros:melodic

RUN mkdir -p /jackal_ws/src
WORKDIR /jackal_ws/src

RUN apt install git


RUN git clone https://github.com/jackal/jackal.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_simulator.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_desktop.git --branch melodic-devel
RUN git clone https://github.com/JvThunder/mlda-barn-2024.git
# RUN git clone https://gitlab.kuleuven.be/u0144428/free_space_motion_tube.git --branch barn2023

WORKDIR /jackal_ws/src/mlda-barn-2024/free_space_motion_tube
RUN mkdir -p build
WORKDIR /jackal_ws/src/mlda-barn-2024/free_space_motion_tube/build
RUN cmake ..
RUN make -j8 && make install

RUN apt-get update && apt-get install -y python3-pip build-essential
RUN pip3 install --upgrade pip
RUN pip3 install defusedxml rospkg netifaces numpy jupyter scipy matplotlib casadi torch scikit-learn pandas easydict

# Install ROS components
RUN apt-get update && apt-get install -y ros-melodic-desktop-full
RUN apt-get install -y ros-melodic-gmapping
RUN apt-get install -y ros-melodic-robot-localization
RUN apt-get install -y ros-melodic-joint-state-publisher-gui
RUN apt-get install -y ros-melodic-navigation
RUN apt-get install -y ros-melodic-hector-gazebo-plugins
RUN apt-get install -y ros-melodic-velodyne-description
RUN apt-get install -y ros-melodic-rosdoc-lite
RUN apt-get install -y ros-melodic-twist-mux
RUN apt-get install -y ros-melodic-sick-tim
RUN apt-get install -y ros-melodic-teleop-twist-joy
RUN apt-get install -y ros-melodic-teleop-twist-keyboard
RUN apt-get install -y ros-melodic-pointgrey-camera-description
RUN apt-get install -y ros-melodic-interactive-marker-twist-server
RUN apt-get install -y ros-melodic-lms1xx
RUN apt-get install -y ros-melodic-laser-pipeline
RUN apt-get install -y ros-melodic-controller-manager
RUN apt-get install -y python-scipy
WORKDIR /jackal_ws

# Add start script
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN echo "if [ ! -d /jackal_ws/devel ]; then ">> ~/.bashrc
RUN echo "  cd /jackal_ws" >> ~/.bashrc
RUN echo "  catkin_make" >> ~/.bashrc
RUN echo "else" >> ~/.bashrc
RUN echo "  source /jackal_ws/devel/setup.bash;" >> ~/.bashrc
RUN echo "fi" >> ~/.bashrc
RUN echo "alias python=python3" >> ~/.bashrc
RUN echo 'cd /jackal_ws/src/mlda-barn-2024' >> ~/.bashrc
WORKDIR /jackal_ws/src/mlda-barn-2024 