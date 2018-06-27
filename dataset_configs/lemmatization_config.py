import models

SEEDS = [5]


RESULTSDIR = '/mnt/storage/bender/projects/cl/sigmorphon-uzh/paper2018/results/lemmatization'  #'../results/lemmatization'


DATASET_SHORTNAME = 'lemmatization'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/lemmatization/data/'


#
DATASET_FOLDS = [
'irish_0_','irish_1_','irish_2_','irish_3_','irish_4_','irish_5_','irish_6_','irish_7_','irish_8_','irish_9_','irish_0_','irish_1_','irish_2_','irish_3_','irish_4_','irish_5_','irish_6_','irish_7_','irish_8_','irish_9_','irish_0_','irish_1_','irish_2_','irish_3_','irish_4_','irish_5_','irish_6_','irish_7_','irish_8_','irish_9_',
'english_0_','english_1_','english_2_','english_3_','english_4_','english_5_','english_6_','english_7_','english_8_','english_9_','english_0_','english_1_','english_2_','english_3_','english_4_','english_5_','english_6_','english_7_','english_8_','english_9_','english_0_','english_1_','english_2_','english_3_','english_4_','english_5_','english_6_','english_7_','english_8_','english_9_',
'basque_0_','basque_1_','basque_2_','basque_3_','basque_4_','basque_5_','basque_6_','basque_7_','basque_8_','basque_9_','basque_0_','basque_1_','basque_2_','basque_3_','basque_4_','basque_5_','basque_6_','basque_7_','basque_8_','basque_9_','basque_0_','basque_1_','basque_2_','basque_3_','basque_4_','basque_5_','basque_6_','basque_7_','basque_8_','basque_9_',
'tagalog_0_','tagalog_1_','tagalog_2_','tagalog_3_','tagalog_4_','tagalog_5_','tagalog_6_','tagalog_7_','tagalog_8_','tagalog_9_','tagalog_0_','tagalog_1_','tagalog_2_','tagalog_3_','tagalog_4_','tagalog_5_','tagalog_6_','tagalog_7_','tagalog_8_','tagalog_9_','tagalog_0_','tagalog_1_','tagalog_2_','tagalog_3_','tagalog_4_','tagalog_5_','tagalog_6_','tagalog_7_','tagalog_8_','tagalog_9_'
                 ]


DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'dev.txt'
DATASET_TESTFILE_NAME = 'test.txt'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [#('hard', '', 'hard'),
                 ('haem', '', 'haem'),
#                ('hacm', '', 'hacm'),
#                 ('haem', '--substitution', 'haem_sub'),
#                 ('hacm', '--copy-as-substitution', 'hacm_sub')
                ]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [#('--align-smart', 'crp'),
                 ('--align-cls', 'cls')
                ]


# the last element in a tuple is the string that goes into the results directory name

MODE_CONFIGS = [#('mle', '', 'mle'),
                #('rl', 'DUMMY', 'rl'),
                #('mrt', 'DUMMY', 'mrt'),
                #('ss', '--pretrain-epochs=0', 'r', 10, 15)
                #models.lols_margin(10, 15),  # seed 1 computed
                models.lols(10, 15),
                #models.dynamic_oracles(10, 15)
               ]


MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 50,
    # DATA
    DATAFORMAT = '--no-feat-format',  # and no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4'
)
