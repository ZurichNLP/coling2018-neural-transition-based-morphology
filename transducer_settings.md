How to Train Transducers on and Apply them to Various Datasets
==========================

## How to use the `lib/launch_dataset.py` script 
This is a generic launcher that needs an appropriate config file in order to compute everything.
Currently, we have the following config files: celex_config.py  ddn13_config.py  nck15_config.py  normalisation_config.py  segmentation_config.py


 0. Make sure that your environment is ok 
 1. `cd lib` # launcher must be called from here

`./launch_dataset.py -h # emits some help and info`

Generate shell commands

 2. `./launch_dataset.py CONFIGFILE > commands.sh  # Outputs the commands on stdout

Execute the commands in parallel on the machine but ignoring all models that have been already successfully built!

 2. `./launch_dataset.py CONFIGFILE -m x -j 20 # -m = Mode of operation x = execute; -j = Parallelization option; use max 20 processes in parallel

The parallelization option should be adapted to the number of available cores on the machine. If you need to stop a running computation from the terminal, it might be
a bit difficult using CTL-C. Type CTL-Z and then use pkill to kill your processes belonging to a specific configuration computation:
`$ pkill -f normalisation_config.py`

Further options: -m e (only work on mle models), -m r (only work on rl/mrt models)

NEW: you can create a Makefile with the corresponding targets and correct dependencies. Allows for more efficient builds as the processes in one chunk do not wait for each othere.
Example code for generating make file for seed one: 

 3. `./launch_dataset.py CONFIGFILE -m m -s 1 > /tmp/CONFIGFILE.mk ; make -f /tmp/CONFIGFILE.mk target -j 20

In order to include special environment initializations before the run_transducer.py command an additional config can be specified that contains a CMDPREFIX variable. For s3it I built the following:
`./launch_dataset.py sgm2017low_config.py s3it_cmd_config.py > commands.sh``

We use the following script to train and evaluate a transducer:


`lib/run_transducer.py`

```
  run_transducer.py [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch ON]
  [--transducer=TRANSDUCER] [--sigm2017format] [--no-feat-format]
  [--input=INPUT] [--feat-input=FEAT] [--action-input=ACTION] [--pos-emb]
  [--enc-hidden=HIDDEN] [--dec-hidden=HIDDEN] [--enc-layers=LAYERS] [--dec-layers=LAYERS]
  [--vanilla-lstm] [--mlp=MLP] [--nonlin=NONLIN] [--lucky-w=W]
  [--pretrain-dropout=DROPOUT] [--dropout=DROPOUT] [--l2=L2]
  [--optimization=OPTIMIZATION] [--batch-size=BATCH-SIZE] [--decbatch-size=BATCH-SIZE]
  [--patience=PATIENCE] [--epochs=EPOCHS] [--pick-loss]
  [--align-smart | --align-dumb | --align-cls] [--tag-wraps=WRAPS] [--try-reverse | --iterations=ITERATIONS]
  [--substitution | --copy-as-substitution] [--param-tying]
  [--mode=MODE] [--verbose] [--beam-width=WIDTH] [--beam-widths=WIDTHS]
  [--pretrain-epochs=EPOCHS | --pretrain-until=ACC] [--sample-size=SAMPLE-SIZE] [--scale-negative=S]
  TRAIN-PATH DEV-PATH RESULTS-PATH [--test-path=TEST-PATH] [--reload-path=RELOAD-PATH]
```

For that, we build the following call:

```python
CALL = """python run_transducer.py --dynet-seed {SEED} --dynet-mem {MEM} --dynet-autobatch {AUTOBATCH} \
  --transducer={TRANSDUCER} {DATAFORMAT} \
  --input={INPUT} --feat-input={FEAT} --action-input={ACTION} {POSEMB} \
  --enc-hidden={EHIDDEN} --dec-hidden={DHIDDEN} --enc-layers={ELAYERS} --dec-layers={DLAYERS} \
  {VANILLALSTM} --mlp={MLP} --nonlin={NONLIN} \
  --dropout={DROPOUT} --optimization={OPTIMIZATION} --l2={L2} \
  --batch-size={BATCHSIZE} --decbatch-size={DECBATCHSIZE} \
  --patience={PATIENCE} --epochs={EPOCHS} {PICKLOSS} \
  {ALIGN} --tag-wraps={WRAPS} --iterations={ITERATIONS} {PARAMTYING} \
  {SUBSTITUTION} --mode={MODE}  {VERBOSE} --beam-width={BEAMWIDTH} --beam-widths={BEAMWIDTHS} \
  {PRETRAIN} --sample-size={SAMPLESIZE} --scale-negative={SNEG} \
  {TRAINPATH} {DEVPATH} {RESULTSPATH} --test-path={TESTPATH} --reload-path={RELOADPATH} 2>&1 | tee {OUTPATH}"""
```

The call can be turned into a shell command with `CALL.format(**params)`, where `params` is
the dictionary with parameters whose keys are `SEED, MEM, AUTOBATCH, etc.`. Below, we define
how `params` is built.

Transducers that we test
==============

* **hard**: hard-attention model of Aharoni & Goldberg
* **haem**: a model with explicit COPY action
* **hacm**: a model with latent copy variable

If we have computing power, we can test:

* **haem-sub**: a haem model with substition actions
* **hacm-sub**: a hacm model with substition actions

All these models are defined by one or maximum two flags

```--transducer={TRANSDUCER} {SUBSTITUTION}```

whose values are the following:

```python
# TRANSDUCER, SUBSTITION, model name
MODEL_CONFIGS = [('hard', '', 'hard'),
                 ('haem', '', 'haem'),
                 ('hacm', '', 'hacm'),
                 ('haem', '--substitution', 'haem_sub'),
                 ('hacm', '--copy-as-substitution', 'hacm_sub')]
```

Oracle alignments that we test
==============================

The options are

* **crp**: Chinese Restaurant Process aligner (aka smart aligner)
* **cls**: Common Longest Substring aligner

The oracle alignment is set by one flag:

```{ALIGN}```

whose values are:

```python
# ALIGN, aligner name
ALIGN_CONFIGS = [('--align-smart', 'crp'), ('--align-cls', 'cls')]
```

Mode
====

We first train all the models with the `mle` criterion. Then we launch `rl` and
`mrt`: These training modes try to improve over the model trained with `mle`,
which gets reloaded in these modes.


Thus, the mode is set with one or two flags:

```python
# MODE, RELOADPATH, mode name
MODE_CONFIGS = [('mle', '', 'mle'),
                 ('rl', reloadpath, 'rl'),
                 ('mrt', reloadpath, 'mrt')]

```
where `reloadpath` is the same as the model's `RESULTPATH` (i.e.
`dataset/model_config/lang/seed`) except that the `mode` of `model_config` must
be changed to `mle`.

**We might want to extend the maximum patience by 5 epochs when training with
`rl` or `mrt`.**


Hyperparameters that we will use
================================

```python

hidden_dim = 200
layers     = 1

PARAMS = dict(
    # DYNET PARAMS
    MEM          = 700,
    AUTOBATCH    = 0,
    # MODEL
    INPUT        = 100,
    FEAT         = 20,
    ACTION       = 100,
    PARAMTYING   = '--param-tying',  # same params for actions & characters
    EHIDDEN      = hidden_dim,
    DHIDDEN      = hidden_dim,
    HIDDEN       = hidden_dim,
    ELAYERS      = layers,
    DLAYERS      = layers,
    LAYERS       = layers,
    VANILLALSTM  = '',  # use CoupledLSTM
    MLP          = 0,   # no hidden-layer in classifier
    NONLIN       = 'ReLU',
    # OPTIMIZATION
    OPTIMIZATION = 'ADADELTA',
    BATCHSIZE    = 1,
    DECBATCHSIZE = 25,  # batchsize used in decoding
    PICKLOSS     = '',  # use dev acc for early stopping
    DROPOUT      = 0,  # no dropout
    PRETRAIN     = '--pretrain-epochs=0',  # skip pretraining    
    # DATA
    WRAPS        = 'both',  # wrap lemma, word with opening & closing boundary tags
    ITERATIONS   = 150,  # for crp aligner
    VERBOSE      = '',
    # NOT USED
    BEAMWIDTH    = 0,  # beam search not used during training
    L2           = 0,   # no l2 regularization
    # RL / MRT
    SAMPLESIZE   = 20,
    SNEG         = 1
)
```

Dataset specific settings
=========================
Each dataset is identified by a string variable DATASET_SHORTNAME with the following values: celex, sgm2016, sgm2017low, sgm2017medium, sgm2017high, nck15, ddn13.

Celex
=====

```python

patience = 10 if mode == 'mle' else 15

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = patience,
    EPOCHS   = 50,
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
```


Sigmorphon 2016
===============

```python

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 30,
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '--pos-emb',
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
```

Wiktionary Dataset (nck15, ddn13)
==================

Let's have the same number of epochs as Aharoni:

* If the training set has more than 50K samples, `epochs=20`,
* If the training set has more than 200K samples, `epochs=5`,
* Otherwise, `epochs=50`.

```python

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 5,
    EPOCHS   = epochs,
    # DATA
    DATAFORMAT = '',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
```

Sigmorphon 2017
===============

Low regime
==========

**Maybe we should try to switch off parameter tying here but not passing "--param-tying".**

```python

patience = 15 if mode == 'mle' else 20  # Different from other datasets!

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = patience,
    EPOCHS   = 60,
    # DATA
    DATAFORMAT = '--sigm2017format',
    POSEMB     = '--pos-emb',
    # DECODING
    BEAMWIDTHS   = '4,8,12,16'  # Different from other datasets!
)
```


Medium regime
============

```python

patience = 10 if mode == 'mle' else 15

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = patience,
    EPOCHS   = 50,
    # DATA
    DATAFORMAT = '--sigm2017format',
    POSEMB     = '--pos-emb',
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
```


High regime
============

```python

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 30,
    # DATA
    DATAFORMAT = '--sigm2017format',
    POSEMB     = '--pos-emb',
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
```


Segmentation
==================

```python

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 10,
    EPOCHS   = 50,
    # DATA
    DATAFORMAT = '--no-feat-format',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
```


Normalization
==================

```python

MORE_PARAMS = dict(
    # OPTIMIZATION
    PATIENCE = 5,
    EPOCHS   = 20,
    # DATA
    DATAFORMAT = '--no-feat-format',  # no --sigm2017format flag
    POSEMB     = '',  # no --pos-emb flag
    # DECODING
    BEAMWIDTHS   = '4,6,8'
)
```
