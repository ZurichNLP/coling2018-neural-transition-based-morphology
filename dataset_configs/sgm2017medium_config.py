

SEEDS = [1, 2, 3, 4, 5]

RESULTSDIR = '../../paper2018/results/sgm2017medium'


DATASET_SHORTNAME = 'sgm2017medium'
# The concatenation of DATASET_PATH/DATASET_FOLD/DATASET_TRAINFILE_NAME gives the filepath
DATASET_PATH = '../data/sigmorphon-task1-2017/data/medium/'

DATASET_FOLDS = [
## launched on idavoll sgm2017medium1.mk
'albanian_',
'arabic_',
'armenian_',
'basque_',
'bengali_',
'bulgarian_',
'catalan_',
'czech_',
'danish_',
'dutch_',
'english_',
'estonian_',
'faroese_',
'finnish_',
'french_',
'georgian_',
'german_',
'haida_',
'hebrew_',
'hindi_',
'hungarian_',
'icelandic_',
'irish_',
'italian_',
'khaling_',
'kurmanji_',
'latin_',
'latvian_',
'lithuanian_',
'lower-sorbian_',
'macedonian_',
'navajo_',
'northern-sami_',
'norwegian-bokmal_',
'norwegian-nynorsk_',
'persian_',
'polish_',
'portuguese_',
'quechua_',
'romanian_',
'russian_',
'scottish-gaelic_',
'serbo-croatian_',
'slovak_',
'slovene_',
'sorani_',
'spanish_',
'swedish_',
'turkish_',
'ukrainian_',
'urdu_',
'welsh_'
]


DATASET_TRAINFILE_NAME = 'train.txt'
DATASET_DEVFILE_NAME = 'dev.txt'
DATASET_TESTFILE_NAME = 'test.txt'

# the last element in a tuple is the string that goes into the results directory name

MODEL_CONFIGS = [('hard', '', 'hard'),
                 ('haem', '', 'haem'),

                ]

# the last element in a tuple is the string that goes into the results directory name

# ALIGN CALL, name:
ALIGN_CONFIGS = [('--align-smart', 'crp'),
                 ('--align-cls', 'cls')
                ]


# the last element in a tuple is the string that goes into the results directory name

MODE_CONFIGS = [
                ('mle', '', ''),
                ('mrt', '--alpha=1', '1', 15, 25),  # with action edit cost
               ]



MORE_PARAMS = dict(
    # OPTIMIZATION
    EPOCHS   = 50,
    #PRETRAIN = '--pretrain-epochs=10',
    # DATA
    DATAFORMAT = '--sigm2017format',
    POSEMB     = '--pos-emb',
    # DECODING
    BEAMWIDTHS   = '4'
)
