
## Use the container (with docker ≥ 19.03)

### Build docker file
```
cd docker
sudo docker build --tag vzc-preprocessing . --no-cache
```
### (optional) push it online
```
sudo docker login
sudo docker tag vzc-preprocessing johnminelli/vzc-preprocessing 
sudo docker push johnminelli/vzc-preprocessing
```
### Create a container from the image
```
sudo docker run --name container -v {local_path_in}:/mounted_input -v {local_path_out}:/mounted_output -dit --rm --gpus all vzc-preprocessing
```
### Execute the container with terminal attached
```
sudo docker exec -it container bash
```
then you can run the preprocessing on your data and training
```
python preprocess.py --action mask --input_path /mounted_input --output_path /mounted_output
python preprocess.py --action pose --input_path /mounted_input --output_path /mounted_output
python preprocess.py --action prompt --input_path /mounted_input --output_path /mounted_output

cd ContextualInpaint/
pip install -r requirements.txt
python diffuser_train.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --output_dir=../<model_store_folder>/ --train_data_dir ../<data_root_folder>/ --num_train_epochs 11 --checkpointing_steps 3000 --gradient_accumulation_steps 4 --train_batch_size 4
```
