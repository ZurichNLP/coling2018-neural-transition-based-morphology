import os
import re

SRC_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(SRC_PATH, '../results')
DATA_PATH = os.path.join(SRC_PATH, '../data/all/task1')
EVALM_PATH = os.path.join(SRC_PATH, '../eval_scripts/evalm.py')

ALIGN_SYMBOL = '~'

### UNK: characters, actions, features
UNK = 0
UNK_CHAR = '#'

### Word boundary: characters, actions
BEGIN_WORD = 1
END_WORD = 2

BEGIN_WORD_CHAR = '<'
END_WORD_CHAR = '>'

### Special actions
STEP = DELETE = 3
COPY = 4
STEP_CHAR = '^'
DELETE_CHAR = '|'
COPY_CHAR   = '='
#STOP_CHAR   = '>'

# all special characters (except feature UNK)
SPECIAL_CHARS = (ALIGN_SYMBOL, BEGIN_WORD_CHAR, END_WORD_CHAR, UNK_CHAR,
                 STEP_CHAR, DELETE_CHAR, COPY_CHAR)

### trainer defaults
MAX_ACTION_SEQ_LEN = 150
SANITY_SIZE = 100

### for docopt argument processing
NULL_ARGS = 'None', 'none', 'no', '0'