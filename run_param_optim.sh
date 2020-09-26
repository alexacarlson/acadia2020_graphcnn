#MESH='sphere' #/root/mesh_data/only_quad_sphere.obj'
#SIL_DEFORM_FLAG=True
#SIL_REF=/root/square_sil.png
#ACOUST_DEFORM_FLAG=True
#WHICH_ACOUST_PARAMS='2660,24920,1.25,2066,17.7,57.7,0.54,0.31,9.37,1.55'
#OUT_NAME='test_sil_deform.obj'
#CFG_PATH='/config/mesh2audioparams_train.yml'

MESH=$1
TRAINED_GRAPH=$2
STYLE_PARAM=$3
SEM_PARAM=$4
FUNC_PARAM=$5
AESTH_PARAM=$6
OUT_NAME=$7
NUM_ITERS=$8
CFG_PATH=$9
#OUT_NAME=${10}
#NUM_ITERS=${11}
#CFG_PATH=${12}

python obj_optim_aestheticparams.py \
  --starting_mesh ${MESH} \
  --trained_graphnet_weights ${TRAINED_GRAPH} \
  --style_param ${STYLE_PARAM} \
  --semantic_param ${SEM_PARAM} \
  --functionvalue_param ${FUNC_PARAM} \
  --aestheticvalue_param ${AESTH_PARAM} \
  --output_filename ${OUT_NAME} \
  --num_iteration ${NUM_ITERS} \
  --config_path ${CFG_PATH}

