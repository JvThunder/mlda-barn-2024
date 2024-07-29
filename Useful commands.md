# Docker

## Build from Dockerfile

```shell
docker build . -t barn:latest --no-cache

docker run --rm -dt --name barn \
	-e DISPLAY=":1" \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $(pwd):/jackal_ws/src/mlda-barn-2024 \
	barn:latest

docker stop <id>
```

## Tag and push to DockerHub

```
docker tag barn:april1 mldarobotics/barn2024:april1
docker push mldarobotics/barn2024:april1
```

## ROS

- Compile ROS Setup
```shell
cd /jackal_ws
catkin_make
source devel/setup.bash
```

- Run one environment
```shell
python run_rviz_kul.py --world_idx 0
```

- Collect data from multiple environments
```shell
./get_kul_data.sh
```