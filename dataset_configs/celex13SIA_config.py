from celex_by_task import (SEEDS, RESULTSDIR, DATASET_PATH, DATASET_FOLDS,
    DATASET_TRAINFILE_NAME, DATASET_DEVFILE_NAME, DATASET_TESTFILE_NAME,
    MODEL_CONFIGS, ALIGN_CONFIGS, MODE_CONFIGS, MORE_PARAMS)

RESULTSDIR += 'celex13SIA'

DATASET_SHORTNAME = 'celex13SIA'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH += '13SIA-13SKE/0500/'