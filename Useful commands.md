# Useful commands for the BARN imitation learning project

## Build from Dockerfile

```shell
docker build . -t barn:latest --no-cache

# Use the following command to build the image with the GPU support
docker run --rm -dt --name barn \
	--gpus all \
	-e DISPLAY=":0" \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $(pwd):/jackal_ws/src/mlda-barn-2024 \
	barn:latest 

# Use the following command to build the image with display support
docker run --rm -dt --name barn \
	-e DISPLAY=":0" \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $(pwd):/jackal_ws/src/mlda-barn-2024 \
	barn:latest 

# Use the following command to build the image without display
docker run --rm -dt --name barn \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $(pwd):/jackal_ws/src/mlda-barn-2024 \
	barn:latest

# Use the following command to stop the currently running container
docker stop <id>
```

## ROS Compilation
```shell
cd /jackal_ws
catkin_make
source devel/setup.bash
```

## Data Collection
- Change the filepath to save in `inspection.py`

- Changing the rate of Lidar from 50Hz to 10Hz
Go to file: `jackal/jackal_description/urdf/hokuyo_ust10.urdf.xacro` and change the following line from 50 to 10
```xml
update_rate:=50
```

- To check the rate of the topics
```shell
rostopic hz /cmd_vel
rostopic hz /front/scan
```

```shell
python run_rviz_kul.py --world_idx 0 # To run KUL on single world_idx
python scripts/get_kul_data.py # To collect data from KUL
```

## Training
- Run the following commands for training the diffusion policy model
```shell
cd train_imitation/diffusion_policy/
pip install -e .
pip install diffusers einops zarr
python /jackal_ws/src/mlda-barn-2024/train_imitation/python/diffusion_policy_model.py
```

- To train the diffusion policy model:
```shell
cd train_imitation/diffusion_policy/
python diffusion_imit_jd.py
```

- To train the behavior cloning model:
Run the notebook `train_imitation/behavior-cloning.ipynb`


## Test on environment
```shell
python run_rviz_kul.py --world_idx 285
python run_rviz_imit.py --world_idx 285
python scripts/eval_imit.py

python scripts/get_data_score.py # To get the score of the imitation model
```

## On progress
- Setting up cuda for GPU support in Docker (check Dockerfile_with_cuda)
- Running the diffusion policy model on the ROS BARN environments
- Evaluating the results of the diffusion policy model