
DATAROOT=/home/alexandracarlson/Desktop/acadia2020_3daesthetics_dataset

docker run -p 9999:9999 --gpus all --shm-size 12G --ipc=host --rm -it \
	-v $DATAROOT:$DATAROOT \
	-v `pwd`:/tf/notebooks/acadia2020_graphcnn_src \
	pytorch3dimg \
	bash -c "jupyter notebook --ip 0.0.0.0 --port 9999 --allow-root" 

