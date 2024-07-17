# Docker

## Build from Dockerfile

```shell
docker build . -t barn:april1 --no-cache

docker run --rm -dt --name barn \
	-e DISPLAY=":1" \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $(pwd):/jackal_ws/src/mlda-barn-2024 \
	barn:april1
```

## Tag and push to DockerHub

```
docker tag barn:april1 mldarobotics/barn2024:april1
docker push mldarobotics/barn2024:april1
```

## ROS

- Clear map

```shell
rosservice call /move_base/clear_costmaps "{}"
```

cd mlda-barn-2024/
python run_rviz_kul.py

cd mlda-barn-2024/
python ./get_kul_data.bash