
#
DATAROOT=/home/alexandracarlson/Desktop/acadia2020_3daesthetics_dataset
NVIDIA_DRIVER_CAPABILITIES=compute,utility 
NVIDIA_VISIBLE_DEVICES=all 
docker run --gpus all --rm -it \
        -v `pwd`:/root \
	-v $DATAROOT:/root/acadia2020_graphcnn_dataset \
	pytorch3dimg \
	python /root/train_aesthetics.py --config-yml /root/config/mesh2aesthetics_train.yml

