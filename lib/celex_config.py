SEEDS = [1, 2, 3, 4, 5]


RESULTSDIR = '/mnt/storage/bender/projects/cl/sigmorphon-uzh/paper2018/results/newcelex'


DATASET_SHORTNAME = 'celex'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/celex/data/'

DATASET_FOLDS = [
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_0.',
	'13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_2.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_1.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_3.',
    '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_4.']




DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'dev.txt'
DATASET_TESTFILE_NAME = 'test.txt'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [('hard', '', 'hard'),
                 ('haem', '', 'haem'),
                 #('hacm', '', 'hacm'),
                 #('haem', '--substitution', 'haem_sub'),
                 #('hacm', '--copy-as-substitution', 'hacm_sub')
                ]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [('--align-smart', 'crp'), #('--align-cls', 'cls')
                ]


# The following MRT settings have producedequally good results (but only on one celex task):
#
# * --epochs=50 --patience=15  --alpha=1 --beta=1 --mode=mrt
# * --epochs=50 --patience=25  --alpha=0.075 --beta=1 --mode=mrt
#
# Notice that the first has a lower max patience.
#
# Also, this now works and seems to improve on MLE results:
#
# --epochs=50 --patience=25 --mode=rl --no-baseline
#
# Interestingly, 1 epoch of idavoll RL training runs 2x slower than 1 epoch of MRT training.
#
# So let's test on celex (celex_config.py and RESULTSDIR = '../../paper2018/results/newcelex') the following conditions:
#
# * -mmrt1    : --mode=mrt --alpha=1
# * -mmrt01   : --mode=mrt --alpha=0.1
# * -mrlnb    : --mode=rl--no-baseline  #SC@PM: Is --no-baseline a separate option?
#
# And let's maybe use more patience!20?

mle_patience = 10
# Lets put patiencesfor mle as 4th and the rl patience as  5th
# # the last element in a tuple is the string that goes into the results directory name
# 1. MODE --mode={MODE}
# 2. MODE_OPTIONS go into the run_transducer.py call as options
# 3. MODEEXTRA goes into the directory name x-...m{MODE}{MODEEXTRA}...-x
# 4. MLE Patience
# 5. Secondary Training Patience (reinforcement patience)
MODE_CONFIGS = [('mrt', '--alpha=0.05', '', mle_patience, 15),
                #('mle', '', '', mle_patience, mle_patience),
                #('rl', ' --no-baseline', 'nb', mle_patience, 20),
                #('rl', '', '', mle_patience, 20),
                #('mrt', '--alpha=0.1', '01', mle_patience, 20), # Do we want to have that? Close to 0.05 we already have?
                #('mrt', '--alpha=1', '1', mle_patience, 20),
                 ]




MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 50,
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4'
)

