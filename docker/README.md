
## Use the container (with docker ≥ 19.03)

```
cd docker/
# Build:
docker build --build-arg USER_ID=$UID -t detectron2:v0 .
# Launch (require GPUs):
docker run --gpus all -it \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --name=detectron2 detectron2:v0

# Grant docker access to host X server to show images
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' detectron2`
```

#### Using a persistent cache directory

You can prevent models from being re-downloaded on every run,
by storing them in a cache directory.

To do this, add `--volume=$HOME/.torch/fvcore_cache:/tmp:rw` in the run command.

## Install new dependencies
Add the following to `Dockerfile` to make persistent changes.
```
RUN sudo apt-get update && sudo apt-get install -y vim
```
Or run them in the container to make temporary changes.


### Build docker file
cd docker
sudo docker build --tag vzc-preprocessing . --no-cache
### (optional) push it online
sudo docker login
sudo docker tag vzc-preprocessing johnminelli/vzc-preprocessing 
sudo docker push johnminelli/vzc-preprocessing
### Create a container from the image
sudo docker run --name container -v /data01/gio/ctrl/a:/mounted_input -v /data01/gio/ctrl/b:/mounted_output -dit --rm --gpus all vzc-preprocessing
### Execute the container with terminal attached
sudo docker exec -it container bash

```
python preprocess.py --action mask --input_path /mounted_input --output_path /mounted_output
python preprocess.py --action pose --input_path /mounted_input --output_path /mounted_output
python preprocess.py --action prompt --input_path /mounted_input --output_path /mounted_output
```