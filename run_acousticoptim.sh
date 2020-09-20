#MESH='sphere' #/root/mesh_data/only_quad_sphere.obj'
#SIL_DEFORM_FLAG=True
#SIL_REF=/root/square_sil.png
#ACOUST_DEFORM_FLAG=True
#WHICH_ACOUST_PARAMS='2660,24920,1.25,2066,17.7,57.7,0.54,0.31,9.37,1.55'
#OUT_NAME='test_sil_deform.obj'
#CFG_PATH='/config/mesh2audioparams_train.yml'

MESH=$1
TRAINED_GRAPH=$2
SIL_DEFORM_FLAG=$3
SIL_REF=$4
SIL_WEIGHT=$5
CAMPOSITION_FILE=$6
ACOUST_DEFORM_FLAG=$7
ACOUST_DEFORM_WEIGHT=$8
WHICH_ACOUST_PARAMS=$9
OUT_NAME=${10}
NUM_ITERS=${11}
CFG_PATH=${12}

python obj_optim_acousticparams.py \
  --which_starting_mesh ${MESH} \
  --trained_graphnet_weights ${TRAINED_GRAPH} \
  --mesh_multisilhouette_optim ${SIL_DEFORM_FLAG} \
  --mesh_acousticparam_optim ${ACOUST_DEFORM_FLAG} \
  --mesh_multisilhouette_optim_weight ${SIL_WEIGHT} \
  --mesh_acousticparam_optim_weight ${ACOUST_DEFORM_WEIGHT} \
  --camera_positions_file ${CAMPOSITION_FILE} \
  --silhouette_img_ref ${SIL_REF} \
  --which_acoustic_params ${WHICH_ACOUST_PARAMS} \
  --output_filename ${OUT_NAME} \
  --num_iteration ${NUM_ITERS} \
  --config_path ${CFG_PATH}

