
#CFG_PATH='config/mesh2audioparams_train.yml'
CFG_PATH=$1
#
NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_VISIBLE_DEVICES=all python train_concert.py --config-yml ${CFG_PATH} 

