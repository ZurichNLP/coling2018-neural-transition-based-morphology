"""Trains and evaluates a state-transition model for inflection generation, using the sigmorphon 2017 shared task
data files and evaluation script.

Usage:
  run_transducer.py [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch ON]
  [--transducer=TRANSDUCER]
  [--input=INPUT] [--feat-input=FEAT] [--action-input=ACTION]
  [--enc-hidden=HIDDEN] [--dec-hidden=HIDDEN] [--enc-layers=LAYERS] [--dec-layers=LAYERS]
  [--vanilla-lstm] [--mlp=MLP] [--nonlin=NONLIN] [--lucky-w=W]
  [--dropout=DROPOUT] [--optimization=OPTIMIZATION] [--batch-size=BATCH-SIZE]
  [--patience=PATIENCE] [--epochs=EPOCHS] [--pick-loss]
  [--align-smart | --align-dumb | --align-cls] [--tag-wraps=WRAPS] [--try-reverse | --iterations=ITERATIONS]
  [--substitution | --copy-as-substitution] [--mode=MODE] [--verbose]
  [--pretrain-epochs=EPOCHS | --pretrain-until=ACC] [--sample-size=SAMPLE-SIZE] [--scale-negative=S]
  TRAIN-PATH DEV-PATH RESULTS-PATH [--test-path=TEST-PATH] [--reload-path=RELOAD-PATH]
  [--beam-width=WIDTH]

Arguments:
  TRAIN-PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV-PATH      development set path, possibly relative to "data/all/"
  RESULTS-PATH  results file to be written, possibly relative to "results"

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET
  --dynet-autobatch ON          perform automatic minibatching
  --transducer=TRANSDUCER       transducer model to use: hacm / haem [default: haem-enc]
  --input=INPUT                 character embedding dimension [default: 100]
  --feat-input=FEAT             feature embedding dimension.  "0" denotes "bag-of-features". [default: 20]
  --action-input=ACTION         action embedding dimension [default: 100]
  --enc-hidden=HIDDEN           hidden layer dimension of encoder RNNs [default: 200]
  --enc-layers=LAYERS           number of layers in encoder RNNs [default: 1]
  --dec-hidden=HIDDEN           hidden layer dimension of decoder RNNs [default: 200]
  --dec-layers=LAYERS           number of layers in decoder RNNs [default: 1]
  --vanilla-lstm                use vanilla LSTM instead of DyNet 1's default coupled LSTM
  --mlp=MLP                     MLP hidden layer dimension. "0" denotes "no hidden layer". [default: 0]
  --nonlin=NONLIN               if mlp, this non-linearity is applied after the hidden layer. ReLU/tanh [default: ReLU]
  --lucky-w=W                   if feat-input==0, scale the "bag-of-features" vector by W [default: 55]
  --dropout=DROPOUT             variotional dropout in decoder RNN [default: 0.5]
  --optimization=OPTIMIZATION   optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: ADADELTA]
  --batch-size=BATCH-SIZE       batch size [default: 1] 
  --patience=PATIENCE           maximal patience for early stopping [default: 10]
  --epochs=EPOCHS               number of training epochs [default: 30]
  --pick-loss                   best model should have the highest dev loss (and not dev accuracy)
  --align-smart                 align with Chinese restaurant process like in Aharoni & Goldberg paper. Default.
  --align-dumb                  align by padding the shortest string on the right (lemma or inflected word)
  --align-cls                   align by aligning the strings' common longest substring first and then padding both strings.
  --try-reverse                 if align-dumb, try reversing lemma and word strings if no COPY action is generated
                                (this will be the case with prefixating morphology)
  --iterations=ITERATIONS       if align-smart, use this number of iterations in the aligner [default: 150]
  --tag-wraps=WRAPS             wrap lemma and word with word boundary tags?
                                  both (use opening and closing tags)/close (only closing tag)/None [default: both]
  --verbose                     visualize results of internal evaluation, display train and dev set alignments
  --substitution                use substitution of y_i (for any x_i) as an action instead of (insert of y_i + delete)
  --copy-as-substitution        treat copy as substitutions?
  --mode=MODE                   various operation modes of the trainer:
                                    eval (run evaluation without training)/mle (MLE training)/rl (reinforcement
                                    learning training) [default: mle]
  --pretrain-epochs=EPOCHS      number of epochs to pretrain the model with MLE training [default: 0]
  --pretrain-until=ACC          MLE pretraining stops as soon as training accuracy score ACC is reached
  --sample-size=SAMPLE-SIZE     if mode==rl, number of samples drawn from the model per training sample [default: 20]
  --scale-negative=S            if mode==rl, scale negative rewards by S [default: 0.1]
  --test-path=TEST-PATH         test set path
  --reload-path=RELOAD-PATH     reload a pretrained model at this path (possibly relative to RESULTS-PATH)
  --beam-width=WIDTH            beam width for beam-search decoding [default: 24]
"""

from __future__ import division
from docopt import docopt

import dynet as dy
import numpy as np
import random
import time

from args_processor import process_arguments
from datasets import BaseDataSet, action2string
from trainer import TrainingSession, internal_eval, dev_external_eval, test_external_eval

if __name__ == "__main__":
    
    np.random.seed(42)
    random.seed(42)
    
    print docopt(__doc__)
    
    print 'Processing arguments...'
    arguments = process_arguments(docopt(__doc__))
    paths, data_arguments, model_arguments, optim_arguments = arguments

    print 'Loading data... Dataset: {}'.format(data_arguments['dataset'])
    train_data = data_arguments['dataset'].from_file(paths['train_path'], **data_arguments)
    VOCAB = train_data.vocab
    VOCAB.train_cutoff()  # knows that entities before come from train set
    dev_data = data_arguments['dataset'].from_file(paths['dev_path'], vocab=VOCAB, **data_arguments)
    if paths['test_path']:
        # no alignments, hence BaseDataSet
        test_data = BaseDataSet.from_file(paths['test_path'], vocab=VOCAB, **data_arguments)
    else:
        test_data = None
        
    batch_size = optim_arguments['batch-size']
    dev_batches = [dev_data.samples[i:i+batch_size] for i in range(0, len(dev_data), batch_size)]

    print 'Building model for training... Transducer: {}'.format(model_arguments['transducer'])
    model = dy.Model()
    transducer = model_arguments['transducer'](model, VOCAB, **model_arguments)
    print 'Trying to load model from: {}'.format(paths['reload_path'])
    model.populate(paths['reload_path'])
    print 'Greedy decoding:'
    accuracy, _, _, _ = internal_eval(dev_batches, transducer, VOCAB, None, False, name='dev')
    print 'Greedy accuracy {}'.format(accuracy)

    print 'Beam-search decoding:'
    then = time.time()
    print 'evaluating on {} data...'.format('dev')

    number_correct = 0.
    total_loss = 0.
    predictions = []
    pred_acts = []
    i = 0  # counter of samples
    for j, batch in enumerate(dev_batches):
        dy.renew_cg()
        batch_loss = []
        for sample in batch: 
            feats = sample.pos, sample.feats
            hypotheses = transducer.beam_search_decode(sample.lemma, feats, external_cg=True,
                                                       beam_width=optim_arguments['beam-width'])
            loss, loss_expr, prediction, predicted_actions = hypotheses[0]
            predictions.append(prediction)
            pred_acts.append(predicted_actions)
            batch_loss.append(loss)
            assert round(loss, 4) == round(loss_expr.scalar_value(), 4), (loss, loss_expr.scalar_value())
            #print 'Target word: ', sample.word_str
            #print
            _, greedy_prediction, _ = transducer.transduce(sample.lemma, feats, external_cg=True)
            #print 'Greedy prediction: ', greedy_prediction

            # evaluation
            correct_prediction = False
            if (prediction in VOCAB.word and VOCAB.word.w2i[prediction] == sample.word):
                correct_prediction = True
                number_correct += 1
                if greedy_prediction != prediction:
                    print 'Beam! Target: ', sample.word_str
                    print 'Greedy prediction: ', greedy_prediction
                    print u'Complete hypotheses:'
                    for log_p, _, pred_word, pred_actions in hypotheses:
                        print u'Actions {}, word {}, -log p {:.3f}'.format(
                            action2string(pred_actions, VOCAB), pred_word, -log_p)

            if False:
                # display prediction for this sample if it differs the prediction
                # of the previous epoch or its an error
                if predicted_actions != previous_predicted_actions[i] or not correct_prediction:
                    #
                    print 'BEFORE:    ', datasets.action2string(previous_predicted_actions[i], VOCAB)
                    print 'THIS TIME: ', datasets.action2string(predicted_actions, VOCAB)
                    print 'TRUE:      ', sample.act_repr
                    print 'PRED:      ', prediction
                    print 'WORD:      ', sample.word_str
                    print 'X' if correct_prediction else 'V'
            # increment counter of samples
            i += 1
        batch_loss = -np.mean(batch_loss)
        total_loss += batch_loss
        # report progress
        if j > 0 and j % 100 == 0: print '\t\t...{} batches'.format(j)

    accuracy = number_correct / i
    print '\t...finished in {:.3f} sec'.format(time.time() - then)
    print 'Beam-{} accuracy {}'.format(optim_arguments['beam-width'], accuracy)