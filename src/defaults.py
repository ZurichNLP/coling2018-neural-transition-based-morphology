import os


# default paths used by scripts like hard_attention.py,
# soft_attention.py, run_all_langs_generic.py
SRC_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(SRC_PATH, '../results')
DATA_PATH = os.path.join(SRC_PATH, '../data/all')

# default values for hard_attention.py
INPUT_DIM = 100
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 100
EPOCHS = 30
LAYERS = 1
OPTIMIZATION = 'ADADELTA'

DYNET_MEM = 2000
DYNET_SEED = 123456
DYNET_SEED_STEP = 100

EVALM_PATH = os.path.join(SRC_PATH, '../eval_scripts/evalm.py')

NBEST = 0 #size of nbest list to print
