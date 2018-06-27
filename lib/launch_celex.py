import multiprocessing
import os
import subprocess
import time

for_real = True
processes_simult = 5
SEEDS = [1, 2, 3, 4, 5]

RESULTSDIR = '/mnt/storage/bender/projects/cl/sigmorphon-uzh/paper2018/results/celex'

CELEX_PATH = '/mnt/storage/hex/users/makarov/morphological-reinflection/data/celex'

CELEX_FOLDS = \
[('13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_0.dev.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_0.test.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_0.train.txt'),
#
 ('13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_1.dev.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_1.test.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_1.train.txt'),
#
 ('13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_2.dev.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_2.test.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_2.train.txt'),
#
 ('13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_3.dev.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_3.test.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_3.train.txt'),
#
('13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_4.dev.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_4.test.txt',
 '13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_4.train.txt')]

CALL = """python run_transducer.py --dynet-seed {SEED} --dynet-mem {MEM} --dynet-autobatch {AUTOBATCH} \
  --transducer={TRANSDUCER} {DATAFORMAT} \
  --input={INPUT} --feat-input={FEAT} --action-input={ACTION} \
  --enc-hidden={EHIDDEN} --dec-hidden={DHIDDEN} --enc-layers={ELAYERS} --dec-layers={DLAYERS} \
  {VANILLALSTM} --mlp={MLP} --nonlin={NONLIN} \
  --dropout={DROPOUT} --optimization={OPTIMIZATION} --l2={L2} \
  --batch-size={BATCHSIZE} --decbatch-size={DECBATCHSIZE} \
  --patience={PATIENCE} --epochs={EPOCHS} {PICKLOSS} \
  {ALIGN} --tag-wraps={WRAPS} --iterations={ITERATIONS} {PARAMTYING} \
  {SUBSTITUTION} --mode={MODE}  {VERBOSE} --beam-width={BEAMWIDTH} --beam-widths={BEAMWIDTHS} \
  {PRETRAIN} --sample-size={SAMPLESIZE} --scale-negative={SNEG} \
  {TRAINPATH} {DEVPATH} {RESULTSPATH} --test-path={TESTPATH} --reload-path={RELOADPATH} 2>&1 | tee {OUTPATH}"""

hidden_dim = 200
layers     = 1

PARAMS = dict(
    MEM          = 700,
    AUTOBATCH    = 0,
    DATAFORMAT   = '', #--sigm2017_format
    INPUT        = 100,
    FEAT         = 20,
    ACTION       = 100,
    EHIDDEN      = hidden_dim,
    DHIDDEN      = hidden_dim,
    HIDDEN       = hidden_dim,
    ELAYERS      = layers,
    DLAYERS      = layers,
    LAYERS       = layers,
    VANILLALSTM  = '',  # use CoupledLSTM
    MLP          = 0,  # no hidden-layer in classifier
    NONLIN       = 'ReLU',
    DROPOUT      = 0,  # no dropout
    OPTIMIZATION = 'ADADELTA',
    BATCHSIZE    = 1,
    DECBATCHSIZE = 25,
    PATIENCE     = 10,
    EPOCHS       = 50,
    PICKLOSS     = '',  # use dev acc for early stopping
    WRAPS        = 'both',
    ITERATIONS   = 150,
    MODE         = 'mrt-beam',  #'rl',
    VERBOSE      = '',
    BEAMWIDTH    = 20,  # for beam-search decoding
    PRETRAIN     = '--pretrain-epochs=0',  # skip pretraining
    SAMPLESIZE   = 20, #1,
    SNEG         = 1,  # not used anyway
    BEAMWIDTHS   = '4,6,8',
    PARAMTYING   = '--param-tying',
    L2           = 0 #0.001
)

# TRANSDUCER, SUBSTITUTION, name:
# * haem (encoder-decoder) with copy action vocab
# * haem (encoder-decoder) with copy action vocab & substitutions
# * hacm (encoder-decoder) with minimal set vocab
# * hacm (encoder-decoder) with copy action vocab & substitutions and where copies are replaced with substitutions
MODEL_CONFIGS = [('haem', '', 'haem'),
                 #('haem', '--substitution', 'haem_sub'),
                 #('hacm', '', 'hacm'),
                 #('hacm', '--copy-as-substitution', 'hacm_sub'),
                 #('hard', '', 'hard')
                 ]

# ALIGN CALL, name:
ALIGN = ('--align-smart', 'crp'), #('--align-cls', 'cls')

RESULTS_DIR = ('x-{MODELNAME}-a{ALIGNNAME}-p{DATASET}-'
               'n{HIDDEN}_{LAYERS}{RNNEXTRA}-w{INPUT}_{FEAT}_{ACTION}{INPEXTRA}-'
               'e{EPOCHS}_{PATIENCE}-o{OPTIMIZATION}_{DROPOUT}{OPTEXTRA}-'
               'm{MODE}{MODEEXTRA}-x')

# EXTRAS
inputextra = '_T' if PARAMS['PARAMTYING'] else ''
optextra = '_l2' if PARAMS['L2'] else ''

calls = []
for FOLD_ID, (dev, test, train) in enumerate(CELEX_FOLDS):
    
    # PATHS
    TRAINPATH = os.path.join(CELEX_PATH, train)
    DEVPATH   = os.path.join(CELEX_PATH, dev)
    TESTPATH  = os.path.join(CELEX_PATH, test)
    
    for TRANSDUCER, SUBSTITUTION, MODELNAME in MODEL_CONFIGS:
        
        for (ALIGNCALL, ALIGNNAME), SEED in ((a, s) for a in ALIGN for s in SEEDS):
            # RESULTS
            RESULTSPATH = os.path.join(RESULTSDIR,
                                       RESULTS_DIR.format(MODELNAME=MODELNAME, ALIGNNAME=ALIGNNAME,
                                                          DATASET='celex',
                                                          RNNEXTRA = '',
                                                          MODEEXTRA = '',
                                                          OPTEXTRA = optextra,
                                                          INPEXTRA=inputextra,
                                                          **PARAMS),
                                       'fold_{}'.format(FOLD_ID),
                                       'seed_{}'.format(SEED))
            
            if PARAMS['MODE'] != 'mle':
                reload_params = dict(PARAMS)
                reload_params['MODE'] = 'mle'
                RELOADPATH = os.path.join(RESULTSDIR,
                                          RESULTS_DIR.format(MODELNAME=MODELNAME, ALIGNNAME=ALIGNNAME,
                                                             DATASET='celex',
                                                             RNNEXTRA = '',
                                                             MODEEXTRA = '',
                                                             OPTEXTRA = optextra,
                                                             INPEXTRA=inputextra,
                                                             **reload_params),
                                          'fold_{}'.format(FOLD_ID),
                                          'seed_{}'.format(SEED))
            else:
                RELOADPATH = ''
            
            if for_real and not os.path.exists(RESULTSPATH):
                os.makedirs(RESULTSPATH)
            
            OUTPATH = os.path.join(RESULTSPATH, 'output.stdout')
            
            call = CALL.format(SEED=SEED,
                               TRANSDUCER=TRANSDUCER,
                               SUBSTITUTION=SUBSTITUTION,
                               ALIGN=ALIGNCALL,
                               TRAINPATH=TRAINPATH,
                               DEVPATH=DEVPATH,
                               TESTPATH=TESTPATH,
                               RESULTSPATH=RESULTSPATH,
                               OUTPATH=OUTPATH,
                               RELOADPATH=RELOADPATH,
                               **PARAMS)
            calls.append(call)

print calls

def launch(call):
    print 'Launching ', call
    if for_real: subprocess.call(call, shell=True)

calls_len = len(calls)
bunches_of_calls = [calls[i:i+processes_simult]
                    for i in range(0, calls_len, processes_simult)]
for i, bunch_of_calls in enumerate(bunches_of_calls):
    then = time.time()
    assert len(bunch_of_calls) <= processes_simult
    pool = multiprocessing.Pool()
    pool.map(launch, bunch_of_calls)
    pool.close()
    pool.join()
    print 'Finished with {} calls out of {}'.format(min((i+1)*processes_simult, calls_len), calls_len)
    now = time.time() - then
    print 'Finished in {:.1f} min'.format(now / 60.)
    print