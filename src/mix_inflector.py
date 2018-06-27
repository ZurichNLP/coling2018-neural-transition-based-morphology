#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains and evaluates a state-transition model for inflection generation, using the sigmorphon 2017 shared task
data files and evaluation script.

Usage:
  mix_inflector_.py [--dynet-seed SEED] [--dynet-mem MEM]
  [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--action-input=ACTION] [--layers=LAYERS]
  [--dropout=DROPOUT] [--second_hidden_layer]
  [--optimization=OPTIMIZATION] [--epochs=EPOCHS] [--patience=PATIENCE]
  [--align_smart | --align_dumb] [--try_reverse] [--iterations=ITERATIONS] [--show_alignments] [--eval]
  TRAIN_PATH DEV_PATH RESULTS_PATH [--test_path=TEST_PATH]

Arguments:
  TRAIN_PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV_PATH      development set path, possibly relative to "data/all/"
  RESULTS_PATH  results file to be written, possibly relative to "results"

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET
  --input=INPUT                 input vector dimensions [default: 100]
  --hidden=HIDDEN               hidden layer dimensions [default: 200]
  --feat-input=FEAT             feature input vector dimension [default: 20]
  --action-input=ACTION         action feature vector dimension [default: 100]
  --layers=LAYERS               amount of layers in LSTMs  [default: 1]
  --dropout=DROPOUT             amount of dropout in LSTMs [default: 0.5]
  --second_hidden_layer         number of FF layers for action prediction from transducer state
  --epochs=EPOCHS               number of training epochs   [default: 30]
  --patience=PATIENCE           patience for early stopping [default: 10]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: ADADELTA]
  --align_smart                 align with Chinese restaurant process like in Aharoni & Goldberg paper. Default.
  --align_dumb                  align by padding the shortest string (lemma or inflected word)
  --try_reverse                 when using dumb alignment, try reversing lemma and word strings
                                if no COPY action is generated (this will the case with prefixating morphology)
  --iterations=ITERATIONS       when using smart alignment, use this number of iterations
                                in the aligner [default: 150]
  --show_alignments             display train and dev set alignments
  --eval                        run evaluation without training
  --test_path=TEST_PATH         test set path
"""
# Differences to hard_mix: features, bow symbol in alignmegnts
from __future__ import division
from docopt import docopt
import os
import sys
import codecs
import random
import progressbar
import time
from collections import Counter, defaultdict

import align
import common
#from hard import HardDataSet, ALIGN_SYMBOL
from defaults import DATA_PATH

import dynet as dy
import numpy as np

from stackLSTM_parser import Vocab, StackRNN
from transition_inflector import (log_to_file, get_accuracy_predictions,
                                  smart_align, dumb_align, ActionDataSet, MAX_ACTION_SEQ_LEN,
                                  OPTIMIZERS, STOP_CHAR, UNK_CHAR, UNK_FEAT_CHAR,
                                  StackBiRNN)

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

ALIGN_SYMBOL = u'~'
#STOP_CHAR   = u'>'
#not needed for mix model, just for checking in main loop - line 444
#DELETE_CHAR = u'|'
#COPY_CHAR   = u'='

STEP_CHAR = u'^'
BEGIN_CHAR = '<'

#UNK_CHAR = '#'
#UNK_FEAT_CHAR = '*'

# defaults for state-transition system
MAX_ACTION_SEQ_LEN = 50

class StackBiRNN_ext(StackBiRNN):
    def __init__(self, *args, **kwargs):
        super(StackBiRNN_ext, self).__init__(*args)
    
    def char(self):
        return self.s[-1][2]

class MixDataSet(ActionDataSet):

    def __init__(self, *args, **kwargs):

        super(MixDataSet, self).__init__(*args)
    
    def build_step_actions(self, lemma, word):
        # returns step action with BEGIN at the start: родич~ - родичи - <^р^о^д^и^чи
        step_actions = []
        lemma = BEGIN_CHAR + lemma
        word = BEGIN_CHAR + word
        for i in range(len(lemma)):
            if word[i]!=ALIGN_SYMBOL:
                step_actions+=word[i]
            if i< len(lemma)-1 and lemma[i+1]!=ALIGN_SYMBOL:
                step_actions+=STEP_CHAR
        return step_actions

    def build_oracle_actions(self, try_reverse=False, verbose=False):

        try_reverse = try_reverse and self.aligner == dumb_align
        if try_reverse:
            print 'USING STRING REVERSING WITH DUMB ALIGNMENT...'
            print 'USING DEFAULT ALIGN SYMBOL ~'

        self.oracle_actions = []
        self.action_set = set()
        for i, (lemma, word) in enumerate(self.aligned_pairs):
            actions = self.build_step_actions(lemma, word)
#            if verbose:
#                print u'{0}\n{1}\n{2}'.format(lemma, u''.join(actions), word)
#                print
            print u'{0}\n{1}\n{2}'.format(lemma, u''.join(actions), word)
            print
            self.oracle_actions.append(actions)
            self.action_set.update(actions)
        self.action_set = sorted(self.action_set)
        print 'finished building oracle actions'
        print 'number of actions: {}'.format(len(self.action_set))

class MixInflector(object):
    def __init__(self, model, train_data, arguments):

        self.INPUT_DIM    = int(arguments['--input'])
        self.HIDDEN_DIM   = int(arguments['--hidden'])
        self.FEAT_INPUT_DIM = int(arguments['--feat-input'])
#        self.ACTION_DIM   = int(arguments['--action-input'])
        self.LAYERS       = int(arguments['--layers'])
        self.dropout      = float(arguments['--dropout'])

        self.build_vocabularies(train_data)
        self.build_model(model)
        # for printing
        self.hyperparams = {'INPUT_DIM'       : self.INPUT_DIM,
                            'HIDDEN_DIM'      : self.HIDDEN_DIM,
                            'FEAT_INPUT_DIM'  : self.FEAT_INPUT_DIM,
#                            'ACTION_INPUT_DIM': self.ACTION_DIM,
                            'LAYERS'          : self.LAYERS,
                            'DROPOUT'         : self.dropout}

    def build_vocabularies(self, train_data):

        # ACTION VOCABULARY
        acts = train_data.action_set + [STOP_CHAR] + [UNK_CHAR]
        self.vocab_acts = Vocab.from_list(acts)

        self.STEP   = self.vocab_acts.w2i[STEP_CHAR]
        self.BEGIN   = self.vocab_acts.w2i[BEGIN_CHAR]
        self.STOP   = self.vocab_acts.w2i[STOP_CHAR]
        self.UNK       = self.vocab_acts.w2i[UNK_CHAR]
        # rest are INSERT_* actions
        INSERT_CHARS, INSERTS = zip(*[(a, a_id) for a, a_id in self.vocab_acts.w2i.iteritems()
                                      if a not in set([STEP_CHAR, BEGIN_CHAR, STOP_CHAR, UNK_CHAR])])

        self.INSERT_CHARS, self.INSERTS = list(INSERT_CHARS), list(INSERTS)
        self.NUM_ACTS = self.vocab_acts.size()
        print u'{} actions of which {} are INSERT actions: {}'.format(self.NUM_ACTS, len(self.INSERTS),
                                                                      u', '.join(self.INSERT_CHARS))
        # FEATURE VOCABULARY
        feats = set([k + u'=' + v for d in train_data.feat_dicts
                     for k, v in d.iteritems()] + [UNK_FEAT_CHAR])
        feat_keys = set([k for d in train_data.feat_dicts
                                  for k, v in d.iteritems()])
        self.vocab_feats = Vocab.from_list(feats)

        self.UNK_FEAT      = self.vocab_feats.w2i[UNK_FEAT_CHAR]
        self.FEATURE_KEYS = list(feat_keys)
        self.NUM_FEATS     = len(feats)
        self.NUM_FEAT_KEYS  = len(feat_keys)
        print '{} features, {} feature keys. Feature keys: {}'.format(self.NUM_FEATS, self.NUM_FEAT_KEYS,
                                                      u', '.join(self.FEATURE_KEYS))

    def build_model(self, model):

        # LSTMs for storing lemma and decoder
        # parameters: layers, in-dim, out-dim, model
        # BiLSTM for lemma
        self.fbuffRNN  = dy.CoupledLSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)
        self.bbuffRNN  = dy.CoupledLSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)
        
        # empty embeddings for all LSTM above
        self.pempty_buffer_emb = model.add_parameters(2*self.HIDDEN_DIM)

        # embedding lookups for characters and actions
        self.ACT_LOOKUP  = model.add_lookup_parameters((self.NUM_ACTS, self.INPUT_DIM))
        self.FEAT_LOOKUP = model.add_lookup_parameters((self.NUM_FEATS, self.FEAT_INPUT_DIM))

        # transducer state to hidden
        # FEATURE VECTOR + 5 LSTMs: 2x lemma hidden representation, 1x actions embedding
#        in_dim = self.HIDDEN_DIM*2 + self.INPUT_DIM + self.NUM_FEATS
        in_dim = self.HIDDEN_DIM*2 + self.INPUT_DIM + self.NUM_FEAT_KEYS*self.FEAT_INPUT_DIM
        
        self.decoderRNN = dy.CoupledLSTMBuilder(self.LAYERS, in_dim, self.HIDDEN_DIM, model)
#        self.pempty_dec    = model.add_parameters(self.HIDDEN_DIM)
        # softmax parameters
        self.R = model.add_parameters((self.NUM_ACTS, self.HIDDEN_DIM))
        self.bias = model.add_parameters(self.NUM_ACTS)
        
        # copy distribution
        self.W_p_gen = model.add_parameters((1, in_dim + self.HIDDEN_DIM))
        self.bias_p_gen = model.add_parameters(1)
        
        print 'Model dimensions:'
        print ' * LEMMA biLSTM (aka BUFFER): IN-DIM: {}, OUT-DIM: {}'.format(2*self.INPUT_DIM,
                                                                             2*self.HIDDEN_DIM)
        print ' * DECODER LSTM:               IN-DIM: {}, OUT-DIM: {}'.format(in_dim, self.HIDDEN_DIM)
        print ' All LSTMs have {} layer(s)'.format(self.LAYERS)
        print
        print ' * ACTION EMBEDDING LAYER:    IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_ACTS, self.INPUT_DIM)
        print
        print ' * SOFTMAX:                   IN-DIM: {}, OUT-DIM: {}'.format(self.HIDDEN_DIM, self.NUM_ACTS)

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def transduce(self, lemma, feats, _oracle_actions=None):

        # encode of oracle actions
        if _oracle_actions:
            oracle_actions = [self.vocab_acts.w2i[a] for a in _oracle_actions[1:]]#exclude self.BEGIN: родич~ - родичи - ^р^о^д^и^чи
            oracle_actions += [self.STOP] #: родич~ - родичи - ^р^о^д^и^чи>
            oracle_actions = list(reversed(oracle_actions))

        # encoding of features: 1 if feature is on, 0 otherwise
#        feats = [k + u'=' + v for k, v in feats.iteritems()]
#        features = np.zeros(self.NUM_FEATS)
#        for f in feats:
#            f_id = self.vocab_feats.w2i.get(f, self.UNK_FEAT)
##            features[f_id] = 55.
#            features[f_id] = 1.
#        features = dy.inputTensor(features)

        feats_keys = [k for k, v in feats.iteritems()]
        features = []
        for f in self.FEATURE_KEYS:
            if f in feats_keys:
                f_str = f + u'=' + feats[f]
                f_id = self.vocab_feats.w2i.get(f_str, self.UNK_FEAT)
                f_embedding = self.FEAT_LOOKUP[f_id]
            else:
                f_embedding = self.FEAT_LOOKUP[self.UNK_FEAT]
            features.append(f_embedding)
        features = dy.concatenate(features)

        buffer = StackBiRNN_ext(self.fbuffRNN, self.bbuffRNN)
        decoder = StackRNN(self.decoderRNN)

        R = dy.parameter(self.R)   # hidden to action
        bias = dy.parameter(self.bias)
        W_p_gen = dy.parameter(self.W_p_gen)   # copy distribution
        bias_p_gen = dy.parameter(self.bias_p_gen)

        # push the characters onto the buffer
        lemma = BEGIN_CHAR + lemma + STOP_CHAR #: <родич~> - родичи - ^р^о^д^и^чи>
        lemma_enc = []
        lemma_emb = []
        for char_ in reversed(lemma):
            char_id = self.vocab_acts.w2i.get(char_, self.UNK)
            char_embedding = self.ACT_LOOKUP[char_id]
            lemma_enc.append((char_embedding, char_))
            lemma_emb.append(char_embedding)
        # char encodings and especially char themselves are put into storage
        # associated with each buffer state.
        buffer.transduce(lemma_emb, lemma_enc)

        losses = []
        word = []
        action_history = [self.BEGIN] # <
        
        while not ((action_history[-1] == self.STOP) or len(action_history) == MAX_ACTION_SEQ_LEN):
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            
            prev_action_id = action_history[-1]
            decoder_input = dy.concatenate([buffer.embedding(), self.ACT_LOOKUP[prev_action_id], features])
            decoder.push(decoder_input)

            probs_gen = dy.softmax(R * decoder.embedding() + bias)
            sigmoid_input = dy.concatenate([buffer.embedding(), decoder.embedding(), self.ACT_LOOKUP[prev_action_id], features])
            p_gen = dy.logistic(W_p_gen * sigmoid_input + bias_p_gen)
            
            _, buffer_char = buffer.char()
            buffer_char_id = self.vocab_acts.w2i.get(buffer_char, self.UNK)
            
            # hack: copy unk symbol from lemma and perform step action
            if buffer_char_id == self.UNK:
#                print buffer_char_id
#                print lemma, u''.join(word), u''.join([self.vocab_acts.i2w[a] for a in action_history[1:]])
                _, insert_char = buffer.pop()
                action = self.STEP
                #losses.append(-dy.log(dy.pick(log_probs, self.UNK)))
                losses.append(-dy.log(dy.pick(probs_gen, self.STEP)))
                action_history.append(action)
                word.append(insert_char)
            else:
                probs_copy_np = np.zeros(self.NUM_ACTS)
                probs_copy_np[buffer_char_id] = 1
                probs_copy = dy.inputTensor(probs_copy_np)

                probs = dy.cmult(p_gen, probs_gen) + dy.cmult(1-p_gen, probs_copy)
                
                
                if _oracle_actions is None:
                    action = np.argmax(probs.npvalue())
                else:
                    action = oracle_actions.pop()
            
                losses.append(-dy.log(dy.pick(probs, action)))
                action_history.append(action)
                # execute the action to update the transducer state
                if action == self.STEP:
                    if buffer_char_id != self.STOP:
                        char_embedding, char_ = buffer.pop()
                elif action == self.STOP:
                    break
                else:
                    # one of the inserts
                    insert_char = self.vocab_acts.i2w.get(action,UNK_CHAR)
                    word.append(insert_char)

        word = u''.join(word)
        action_history = u''.join([self.vocab_acts.i2w[a] for a in action_history[1:]])
#        if dy.average(losses).npvalue() > 500:
#            print lemma, word, action_history, [loss.npvalue() for loss in losses]
        return ((dy.average(losses) if losses else None), word, action_history)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments

    train_path        = common.check_path(arguments['TRAIN_PATH'], 'TRAIN_PATH')
    dev_path          = common.check_path(arguments['DEV_PATH'], 'DEV_PATH')
    results_file_path = common.check_path(arguments['RESULTS_PATH'], 'RESULTS_PATH', is_data_path=False)

    # some filenames defined from `results_file_path`
    log_file_name   = results_file_path + '_log.txt'
    tmp_model_path  = results_file_path + '_bestmodel.txt'

    if arguments['--test_path']:
        test_path = common.check_path(arguments['--test_path'], 'test_path')
    else:
        # indicates no test set eval should be performed
        test_path = None

    print 'Train path: {}'.format(train_path)
    print 'Dev path: {}'.format(dev_path)
    print 'Results path: {}'.format(results_file_path)
    print 'Test path: {}'.format(test_path)

    lang, _, regime = os.path.basename(train_path).rsplit('-', 2)
    print 'LANGUAGE: {}, REGIME: {}'.format(lang, regime)

    print 'Loading data...'
    train_data = MixDataSet.from_file(train_path)
    dev_data = MixDataSet.from_file(dev_path)
    if test_path:
        test_data = MixDataSet.from_file(test_path)
    else:
        test_data = None

    print 'Checking if any special symbols in data...'
    for data, name in [(train_data, 'train'), (dev_data, 'dev')] + \
        ([(test_data, 'test')] if test_data else []):
        data = set(data.lemmas + data.words)  # for test words are 'COVER' ..?
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR, STEP_CHAR]:
            assert c not in data
        print '{} data does not contain special symbols'.format(name)

    is_dumb = arguments['--align_dumb']
    aligner = dumb_align if is_dumb else smart_align
    try_reverse = arguments['--try_reverse']
    show_alignments = arguments['--show_alignments']
    align_iterations = int(arguments['--iterations'])
    train_data.align_and_build_actions(aligner=aligner,
                                       try_reverse=try_reverse,
                                       verbose=show_alignments,
                                       iterations=align_iterations)

    dev_data.align_and_build_actions(aligner=aligner,
                                     try_reverse=try_reverse,
                                     verbose=show_alignments,
                                     iterations=align_iterations)
    print 'Building model...'
    model = dy.Model()
    ti = MixInflector(model, train_data, arguments)

    optimization = arguments['--optimization']
    epochs = int(arguments['--epochs'])
    max_patience = int(arguments['--patience'])

    hyperparams = {'ALIGNER': 'DUMB' if is_dumb else 'SMART',
                   'ALIGN_ITERATIONS': align_iterations if not is_dumb else 'does not apply',
                   'TRY_REVERSE': try_reverse if is_dumb else 'does not apply',
                   'MAX_ACTION_SEQ_LEN': MAX_ACTION_SEQ_LEN,
                   'OPTIMIZATION': optimization,
                   'EPOCHS': epochs,
                   'PATIENCE': max_patience,
                   'BEAM_WIDTH': 1,
                   'FEATURE_TYPE_DETECT': None}
    for k, v in ti.hyperparams.items():
        hyperparams[k] = v

    for k, v in hyperparams.items():
        print '{:20} = {}'.format(k, v)
    print

    if not arguments['--eval']:
        # perform model training
        trainer = OPTIMIZERS.get(optimization, OPTIMIZERS['SGD'])
        print 'Using {} trainer: {}'.format(optimization, trainer)
        trainer = trainer(model)

        total_loss = 0.  # total train loss that is...
        best_avg_dev_loss = 999.
        best_dev_accuracy = -1.
        best_train_accuracy = -1.
        train_len = train_data.length
        dev_len = dev_data.length
        sanity_set_size = 100
        patience = 0
        previous_predicted_actions = [[None]*sanity_set_size]

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
        avg_loss = -1  # avg training loss that is...

        # does not change from epoch to epoch due to re-shuffling
        dev_set = dev_data.iter()

        for epoch in xrange(epochs):
            print 'training...'
            then = time.time()
            # ENABLE DROPOUT
            #ti.set_dropout()
            # compute loss for each sample and update
            for i, (lemma, word, actions, feats) in enumerate(train_data.iter(shuffle=True)):
                # here we do training
                dy.renew_cg()
                loss, prediction, _ = ti.transduce(lemma, feats, actions)
                #print lemma, word, prediction
                if loss is not None:
                    total_loss += loss.scalar_value()
                    loss.backward()
                    trainer.update()
                if i > 0:
                    avg_loss = total_loss / (i + epoch * train_len)
                else:
                    avg_loss = total_loss
            # DISABLE DROPOUT AFTER TRAINING
            #ti.disable_dropout()
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            # condition for displaying stuff like true words and predictions
            check_condition = (epoch > 0 and (epoch % 5 == 0 or epoch == epochs - 1))

            # get train accuracy
            print 'evaluating on train...'
            then = time.time()
            train_correct = 0.
            pred_acts = []
            for i, (lemma, word, actions, feats) in enumerate(train_data.iter(indices=sanity_set_size)):
                _, prediction, predicted_actions = ti.transduce(lemma, feats)
#                print lemma, word, prediction
                pred_acts.append(predicted_actions)

                if check_condition:
                    if predicted_actions != previous_predicted_actions[-1][i]:
                        print 'BEFORE:    ', previous_predicted_actions[-1][i]
                        print 'THIS TIME: ', predicted_actions
                        print 'TRUE:      ', u''.join(actions) + STOP_CHAR
                        print 'PRED:      ', prediction
                        print 'WORD:      ', word
                        print

                if prediction == word:
                    train_correct += 1

            previous_predicted_actions.append(pred_acts)
            train_accuracy = train_correct / sanity_set_size
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()
            dev_correct = 0.
            dev_loss = 0.
            for lemma, word, actions, feats in dev_set:
                loss, prediction, predicted_actions = ti.transduce(lemma, feats)
                if prediction == word:
                    dev_correct += 1
                    tick = 'V'
                else:
                    tick = 'X'
#                if check_condition:
#                    print 'TRUE:    ', word
#                    print 'PRED:    ', prediction
#                    print tick
#                    print
                dev_loss += loss.scalar_value()
            dev_accuracy = dev_correct / dev_len
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                #then = time.time()
                # save best model to disk
                model.save(tmp_model_path)
                print 'saved new best model to {}'.format(tmp_model_path)
                #print '\t...finished in {:.3f} sec'.format(time.time() - then)
                patience = 0
            else:
                patience += 1

            # found "perfect" model
            if dev_accuracy == 1:
                train_progress_bar.finish()
                break

            # get dev loss
            avg_dev_loss = dev_loss / dev_len

            if avg_dev_loss < best_avg_dev_loss:
                best_avg_dev_loss = avg_dev_loss

            print ('epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} '
                   'train accuracy: {4:.4f} best dev accuracy: {5:.4f} best train accuracy: {6:.4f} '
                   'patience = {7}').format(epoch, avg_loss, avg_dev_loss, dev_accuracy, train_accuracy,
                                            best_dev_accuracy, best_train_accuracy, patience)

            log_to_file(log_file_name, epoch, avg_loss, train_accuracy, dev_accuracy)

            if patience == max_patience:
                print 'out of patience after {} epochs'.format(epoch)
                train_progress_bar.finish()
                break
            # finished epoch
            train_progress_bar.update(epoch)
        print 'finished training. average loss: {}'.format(avg_loss)

    else:
        print 'skipped training by request. evaluating best models.'

    # eval on dev
    print '=========DEV EVALUATION:========='
    model = dy.Model()
    ti = MixInflector(model, train_data, arguments)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.populate(tmp_model_path)

    accuracy, dev_results = get_accuracy_predictions(ti, dev_data)
    print 'accuracy: {}'.format(accuracy)
    output_file_path = results_file_path + '.best'

    common.write_results_file_and_evaluate_externally(hyperparams, accuracy,
                                                      train_path, dev_path, output_file_path,
                                                      # nbest=True simply means inflection comes as a list
                                                      {i : v for i, v in enumerate(dev_results)},
                                                      nbest=True, test=False)
    if test_data:
        # eval on test
        print '=========TEST EVALUATION:========='
        accuracy, test_results = get_accuracy_predictions(ti, test_data)
        print 'accuracy: {}'.format(accuracy)
        output_file_path = results_file_path + '.best.test'

        common.write_results_file_and_evaluate_externally(hyperparams, accuracy,
                                                          train_path, test_path, output_file_path,
                                                          # nbest=True simply means inflection comes as a list
                                                          {i : v for i, v in enumerate(test_results)},
                                                          nbest=True, test=True)
