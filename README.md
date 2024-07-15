
# MLDA_EEE



https://github.com/MLDA-NTU/mlda-barn-2024/assets/32756835/d21bdda6-0903-49c4-aa32-c9ca440c276b



This is the repository for the proposed solution of Team MLDA_EEE (Machine Learning & Data Analytics) from Nanyang Technological University, Singapore

The original readme file is in [README_BARN.md](./README_BARN.md)

# Navigation Stack

Launch file at `./jackal_helper/launch/move_base_mlda.launch`
Source files at `./mlda_algo`

We also create a `run_rviz.py` to launch `move_base_mlda_rviz.launch` to visualize and troubleshoot

# Container Environment

## Singularity Image

We use Go 1.20 and Singularity 4.0.2

```shell
# Build image name 'nav_competition_image.sif'
sudo singularity build --notest nav_competition_image.sif Singularityfile.def


# Run
./singularity_run.sh ./nav_competition_image.sif python3 run.py --world_idx 0
```

- On Ubuntu 18.04 Machine

We can build and run the Singularity image

- On Ubuntu 22.04 Machine

We can build the Singularity image but it cannot execute the `run.py` program through it due to `GLIBC=2.34 missing` error and we cannot fix. In the `.def` file, we also installed all the necessary `ros-melodic-*` packages instead of relying on `rosdep`

## Docker Image

This is the docker image for the April 1st Soft-Deadline submission

```shell
docker pull mldarobotics/barn2024:april1
```

Start the docker container named `barn` in the background. Use VSCode `Dev Container` extension to connect to the running container

```shell
# Allow GUI to be displayed
xhost +

# No Nvidia Container
docker run --rm -dt --name barn \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  mldarobotics/barn2024:april1


# Nvidia Container
docker run --rm -dt --name barn \
  --gpus all \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  mldarobotics/barn2024:april1
```

In the docker, the folder structure is similar to the suggested folder structure in the original README_BARN.md

```shell
# Navigate to this folder in the container
cd /jackal_ws/src/mlda-barn-2024/

# standard
python3 run.py --world_idx 0 --gui

# with rviz
python3 run_rviz.py --world_idx 0 --gui
```

# Gazebo World

All of 300 worlds `.world` files have their initial camera angle changed for ease of viewing and troubleshooting

```xml
<gui fullscreen="0">
  <camera name="user_camera">
    <pose frame="">-2 4 10 0 1.57 3.14</pose>
    <view_controller>orbit</view_controller>
    <projection_type>perspective</projection_type>
  </camera>
</gui>
```

DYNABarn is not used
