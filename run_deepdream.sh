MESH='plane' #/root/mesh_data/only_quad_sphere.obj'
OUT_NAME='test_dream.obj'
WHICH_LAYER='gconv1'
WHICH_VERT=5
WHICH_FEAT=100
CFG_PATH='/root/config/style_transfer_train.yml'
#
#   --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
docker run --rm -it -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
  --gpus all --shm-size 8G \
  -v `pwd`:/root \
  -it pytorch3d-img \
  python /root/deepdream.py \
  --which_starting_mesh ${MESH} \
  --which_feature ${WHICH_FEAT}\
  --which_layer ${WHICH_LAYER} \
  --which_vertex ${WHICH_VERT}\
  --output_filename ${OUT_NAME} \
  --config_path ${CFG_PATH} 

