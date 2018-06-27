"""Trains and evaluates a state-transition model for inflection generation, using the sigmorphon 2017 shared task
data files and evaluation script.

Usage:
  transition_inflector_.py [--dynet-seed SEED] [--dynet-mem MEM]
  [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--action-input=ACTION] [--layers=LAYERS]
  [--dropout=DROPOUT] [--second_hidden_layer]
  [--optimization=OPTIMIZATION] [--epochs=EPOCHS] [--patience=PATIENCE] [--pretrain_copy=EPOCHS]
  [--align_smart | --align_dumb] [--try_reverse] [--iterations=ITERATIONS] [--show_alignments] [--eval] [--samples=SAMPLES] [--pretrain_epochs=PRETRAIN_EPOCHS]
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
  --feat-input=FEAT             feature input vector dimension; CURRENTLY INGORED
  --action-input=ACTION         action feature vector dimension [default: 100]
  --layers=LAYERS               amount of layers in LSTMs  [default: 1]
  --dropout=DROPOUT             amount of dropout in LSTMs [default: 0.5]
  --second_hidden_layer         number of FF layers for action prediction from transducer state
  --epochs=EPOCHS               number of training epochs   [default: 30]
  --patience=PATIENCE           patience for early stopping [default: 10]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: ADADELTA]
  --pretrain_copy=EPOCHS        number of epochs to pretrain on copy data built from train data [default: 0]
  --align_smart                 align with Chinese restaurant process like in Aharoni & Goldberg paper. Default.
  --align_dumb                  align by padding the shortest string (lemma or inflected word)
  --try_reverse                 when using dumb alignment, try reversing lemma and word strings
                                if no COPY action is generated (this will the case with prefixating morphology)
  --iterations=ITERATIONS       when using smart alignment, use this number of iterations
                                in the aligner [default: 150]
  --show_alignments             display train and dev set alignments
  --eval                        run evaluation without training
  --test_path=TEST_PATH         test set path
  --samples=SAMPLES             how many samples to use in reinforcement learning
  --pretrain_epochs=PRETRAIN_EPOCHS number epochs for pretraining in reinforcement learning TODO
"""

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
from hard import HardDataSet, ALIGN_SYMBOL
from defaults import DATA_PATH

import dynet as dy
import numpy as np

from stackLSTM_parser import Vocab, StackRNN

from transition_inflector import ActionDataSet, TransitionInflector, TwoLayerTransitionInflector, smart_align, dumb_align, infer_lemma_feats, log_to_file, get_accuracy_predictions
from NNAligner import NNAligner, align_pair

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


STOP_CHAR   = u'>'
DELETE_CHAR = u'|'
COPY_CHAR   = u'='

UNK_CHAR = '#'
UNK_FEAT_CHAR = '*'

# defaults for state-transition system
MAX_ACTION_SEQ_LEN = 50

OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, lam=0.0, alpha=0.0001,
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
              'MOMENTUM': dy.MomentumSGDTrainer,
              'SGD'     : dy.SimpleSGDTrainer,
              'ADAGRAD' : dy.AdagradTrainer,
              'ADADELTA': dy.AdadeltaTrainer}


class FullActionDataSet(ActionDataSet):

    def __init__(self, *args, **kwargs):
        
        super(FullActionDataSet, self).__init__(*args)

    def build_oracle_actions(self, try_reverse=False, verbose=False):
        
        try_reverse = try_reverse and self.aligner == dumb_align
        if try_reverse:
            print 'USING STRING REVERSING WITH DUMB ALIGNMENT...'
            print 'USING DEFAULT ALIGN SYMBOL ~'

        self.oracle_actions = []
        self.action_set = set()
        for i, (lemma, word) in enumerate(self.aligned_pairs):
            code, true_code = self.build_code(lemma, word)
            
            if try_reverse and COPY_CHAR not in code:
                # no copying is being done, probably
                # this sample uses prefixation. Try aligning
                # from the end:
                pair = self.lemmas[i][::-1], self.words[i][::-1]
                [(new_al_lemma, new_al_word)] = self.aligner([pair], ALIGN_SYMBOL)
                rcode, rtrue_code = build_code(new_al_lemma[::-1], new_al_word[::-1])
                if COPY_CHAR in rcode:
                    print u'Reversed aligned: {} => {}'.format(lemma, word)
                    print (u'Forward alignment: {}, '
                           'REVERSED alignment: {}'.format(u''.join(code),
                                                           u''.join(rcode)))
                    code = rcode
                    true_code = rtrue_code
        
            actions = self.build_edit_actions(code)
            
            if verbose:
                print u'{0}\n{1}\n{2}\n{3}'.format(lemma, u''.join(actions),
                                                   u''.join(true_code), word)
                print
            self.oracle_actions.append(actions)
            self.action_set.update(actions)
            self.action_set.update([c for c in lemma]+[c for c in word]) # !!!
        self.action_set = sorted(self.action_set)
        print 'finished building oracle actions'
        print 'number of actions: {}'.format(len(self.action_set))



if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments

    train_path        = common.check_path(arguments['TRAIN_PATH'], 'TRAIN_PATH')
    dev_path          = common.check_path(arguments['DEV_PATH'], 'DEV_PATH')
    results_file_path = common.check_path(arguments['RESULTS_PATH'], 'RESULTS_PATH', is_data_path=False)

    # some filenames defined from `results_file_path`
    log_file_name   = results_file_path + '_log.txt'
    tmp_model_path  = results_file_path + '_bestmodel.txt'
    tmp_model_path_align  = results_file_path + '_bestmodel_align.txt'
    tmp_model_path_pretrain  = results_file_path + '_bestmodel_pretrain.txt'
    tmp_model_path_align_pretrain  = results_file_path + '_bestmodel_align_pretrain.txt'
    

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
    train_set = FullActionDataSet.from_file(train_path)
    dev_set = FullActionDataSet.from_file(dev_path)
    if test_path:
        test_set = FullActionDataSet.from_file(test_path)
    else:
        test_set = None

    print 'Checking if any special symbols in data...'
    for data, name in [(train_set, 'train'), (dev_set, 'dev')] + \
        ([(test_set, 'test')] if test_set else []):
        data = set(data.lemmas + data.words)  # for test words are 'COVER' ..?
        for c in [COPY_CHAR, DELETE_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print '{} data does not contain special symbols'.format(name)

    is_dumb = arguments['--align_dumb']
    aligner = dumb_align if is_dumb else smart_align
    try_reverse = arguments['--try_reverse']
    show_alignments = arguments['--show_alignments']
    align_iterations = int(arguments['--iterations'])


    train_set.align_and_build_actions(aligner=aligner,
                                       try_reverse=try_reverse,
                                       verbose=show_alignments,
                                       iterations=align_iterations)


    dev_set.align_and_build_actions(aligner=aligner,
                                     try_reverse=try_reverse,
                                     verbose=show_alignments,
                                     iterations=align_iterations)


    print 'Building transition model...'
    model = dy.Model()
    if arguments['--second_hidden_layer']:
        ti = TwoLayerTransitionInflector(model, train_set, arguments)
    else:
        ti = TransitionInflector(model, train_set, arguments)

    print 'Building alignment model...'
    model_A = dy.Model()
    al = NNAligner(model_A, train_set) # TODO parameter arguments?


    optimization = arguments['--optimization']
    epochs = int(arguments['--epochs'])
    max_patience = int(arguments['--patience'])
    if arguments['--pretrain_epochs']:
        pretrain_epochs = int(arguments['--pretrain_epochs']) #!!!
    else:
        pretrain_epochs = None
    samples = int(arguments['--samples']) # !!!

    hyperparams = {'ALIGNER': 'DUMB' if is_dumb else 'SMART',
                   'ALIGN_ITERATIONS': align_iterations if not is_dumb else 'does not apply',
                   'TRY_REVERSE': try_reverse if is_dumb else 'does not apply',
                   'MAX_ACTION_SEQ_LEN': MAX_ACTION_SEQ_LEN,
                   'OPTIMIZATION': optimization,
                   'EPOCHS': epochs,
                   'PATIENCE': max_patience,
                   'BEAM_WIDTH': 1,
                   'FEATURE_TYPE_DETECT': None,
                   'PRETRAINING EPOCHS': pretrain_epochs,
                   'SAMPLES':samples}
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


        trainer_A = OPTIMIZERS.get(optimization, OPTIMIZERS['SGD'])
        print 'Using {} trainer: {}'.format(optimization, trainer_A)
        trainer_A = trainer_A(model_A)

        # Tracking variables
        total_loss = 0.  # total train loss that is...
        best_avg_dev_loss = 999.
        best_dev_accuracy = -1.
        best_train_accuracy = -1.
        train_len = train_set.length
        dev_len = dev_set.length
        sanity_set_size = 100
        patience = 0
        previous_predicted_actions = [[None]*sanity_set_size]

        total_loss_A = 0.  # total train loss that is...
        best_avg_dev_loss_A = 999.
        best_dev_accuracy_A = -1.
        best_train_accuracy_A = -1.


        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
        avg_loss = -1  # avg training loss that is...

        avg_loss_A = -1
        
        np.random.seed(123)

        random.seed(123)
        
        ##### PRETRAIN #####
        
        if pretrain_epochs:
            # pretrain data
            align_actions = al.get_labels(aligner, zip(train_set.lemmas, train_set.words))
            pretrain_data = zip(train_set.lemmas, train_set.words, train_set.oracle_actions, train_set.feat_dicts, align_actions)
            dev_data = zip(dev_set.lemmas, dev_set.words, dev_set.feat_dicts)
            
            for epoch in xrange(pretrain_epochs):
                print 'Supervised pretraining...'
                then = time.time()
                # ENABLE DROPOUT
                ti.set_dropout()
                # compute loss for each sample and update
                np.random.shuffle(pretrain_data)
                for i, (lemma, word, edit_actions, feats, align_actions) in enumerate(pretrain_data):
                    
                    loss_B, _, _ = ti.transduce(lemma, feats, edit_actions)
                    if loss_B is not None:
                        total_loss += loss_B.scalar_value()
                        loss_B.backward()
                        trainer.update()
                    if i > 0:
                        avg_loss = total_loss / (i + epoch * train_len)
                    else:
                        avg_loss = total_loss
            
                    if epoch < 3: # TODO: maybe separate loop? maybe add as a parameter
                    
                        _, loss_A = al.align((lemma, word), align_actions)
                        if loss_A is not None:
                            total_loss_A += loss_A.scalar_value()
                            loss_A.backward()
                            trainer_A.update()
                        if i > 0:
                            avg_loss_A = total_loss_A / (i + epoch * train_len)
                        else:
                            avg_loss_A = total_loss_A
                # DISABLE DROPOUT AFTER TRAINING
                ti.disable_dropout()
                # predict train set
                correct = 0.
                correct_A = 0.
                for j, (lemma, word, edit_actions, feats, align_actions) in enumerate(pretrain_data[:100]):

                    _, prediction, _ = ti.transduce(lemma, feats)
                    if prediction == word:
                        correct += 1
#                    else:
#                        print 'TRUE:    ', word
#                        print 'PRED:    ', prediction
#                        print 'X'

                    predicted_actions, _ = al.align((lemma,word))

                    L, F = align_pair(predicted_actions, lemma, word)
                    if predicted_actions == align_actions:
                        correct_A += 1
#                    else:
#                        print L, F
#                        print 'TRUE:    ', align_actions
#                        print 'PRED:    ', predicted_actions
#                        print 'X'

                accuracy = correct / 100
                accuracy_A = correct_A / 100
                print 'Pretraining: epoch {}, sanity accuracy Model B {}, train loss Model B {}, sanity accuracy Model A {}, train loss Model A {}'.format(epoch, accuracy, avg_loss, accuracy_A, avg_loss_A)
                
            print 'Finished pretraining...'

            model.save(tmp_model_path_pretrain)
            print 'saved new best transition model to {}'.format(tmp_model_path_pretrain)
            model_A.save(tmp_model_path_align_pretrain)
            print 'saved new best alignment model to {}'.format(tmp_model_path_align_pretrain)

        else:
            print 'trying to load transition model from: {}'.format(tmp_model_path_pretrain)
            model.populate(tmp_model_path_pretrain)
            print 'trying to load alignment model from: {}'.format(tmp_model_path_align_pretrain)
            model_A.populate(tmp_model_path_align_pretrain)
    
    
    
        ##### TRAIN #####
        
        # train data
        train_data = zip(train_set.lemmas, train_set.words, train_set.feat_dicts)
        dev_data = zip(dev_set.lemmas, dev_set.words, dev_set.feat_dicts)
        # Tracking variables
        total_loss = 0.  # total train loss that is...
        best_avg_dev_loss = 999.
        best_dev_accuracy = -1.
        best_train_accuracy = -1.
        train_len = train_set.length
        dev_len = dev_set.length
        sanity_set_size = 100
        patience = 0
        previous_predicted_actions = [[None]*sanity_set_size]
        
        total_loss_A = 0.  # total train loss that is...
        best_avg_dev_loss_A = 999.
        best_dev_accuracy_A = -1.
        best_train_accuracy_A = -1.

        # get dev accuracy for model B, the starting point for tracking improvements
        print 'evaluating Model B on dev...'
        then = time.time()
        dev_correct = 0.
        dev_loss = 0.
        for lemma, word, feats in dev_data:
            loss, prediction, predicted_actions = ti.transduce(lemma, feats)
            if prediction == word:
                dev_correct += 1
            dev_loss += loss.scalar_value()
        best_dev_accuracy = dev_correct / dev_len
        print '\t...finished in {:.3f} sec'.format(time.time() - then)
        print 'Pretrained Model B: dev accuracy {} dev loss {}'.format(best_dev_accuracy, dev_loss)
        # save pretrained models as best models
        model.save(tmp_model_path)
        print 'saved pretrained transition model as new best to {}'.format(tmp_model_path)
        model_A.save(tmp_model_path_align)
        print 'saved alignment model as new best to {}'.format(tmp_model_path_align)


        for epoch in xrange(epochs):
            print 'Reinforce training...'
            then = time.time()
            np.random.shuffle(train_data)
            
            # Update Model A
            for i, (lemma, word, feats) in enumerate(train_data):
                dy.renew_cg()
                loss_A = []
                for j in xrange(samples):
                    predicted_actions, loss = al.align((lemma, word), sampling=True, external_cg=True)
                    L, F = align_pair(predicted_actions, lemma, word)
                    code, true_code = train_set.build_code(L, F)
                    edit_actions = train_set.build_edit_actions(code)
                    # now both models contribute to CG!!!
                    loss_B, _, _ = ti.transduce(lemma, feats, edit_actions, external_cg=True)
                    loss_A.append(dy.scalarInput(np.exp(-loss_B.value())) * loss)
                loss_A = dy.esum(loss_A)
                total_loss_A += loss_A.scalar_value()
                loss_A.backward()
                trainer_A.update()
                if i > 0:
                    avg_loss_A = total_loss_A / (i + epoch * train_len)
                else:
                    avg_loss_A = total_loss_A

            print '\t...finished in {:.3f} sec'.format(time.time() - then)
            print 'epoch: {} train loss Model A: {}'.format(epoch, avg_loss_A)
                        
            if epoch % 3==0 and epoch > 0:
            # Update Model B every 3 epochs
                #ti.set_dropout()
                # compute loss for each sample and update
#                edit_actions_current = [None]*train_len
                for i, (lemma, word, feats) in enumerate(train_data):

                    predicted_actions, _ = al.align((lemma, word))
                    L, F = align_pair(predicted_actions, lemma, word)
                    code, true_code = train_set.build_code(L, F)
                    edit_actions = train_set.build_edit_actions(code)
#                    edit_actions_current[i] = edit_actions

                    loss_B, _, _ = ti.transduce(lemma, feats, edit_actions)
                    if loss_B is not None: #?
                        total_loss += loss_B.scalar_value()
                        loss_B.backward()
                        trainer.update()
                    if i > 0:
                        avg_loss = total_loss / (i + epoch * train_len)
                else:
                    avg_loss = total_loss
                
                print 'epoch: {} train loss Model B: {}'.format(epoch, avg_loss)

                #ti.disable_dropout()

                # Check the train and dev accuracy and save the models if dev accuracy increases:
                
                # condition for displaying stuff like true words and predictions
                check_condition = (epoch > 0 and (epoch % 5 == 0 or epoch == epochs - 1))

                # get train accuracy
                print 'evaluating on train...'
                then = time.time()
                train_correct = 0.
#                pred_acts = []
                for i, (lemma, word, feats) in enumerate(train_data[:100]):
                    #print lemma, word, feats, edit_actions
<<<<<<< Updated upstream
                    _, prediction, predicted_actions = ti.transduce(lemma, feats)
#                    pred_acts.append(predicted_actions)
=======
                    
                    _, prediction, predicted_actions = ti.transduce(lemma, feats, edit_actions_current[i])
                    pred_acts.append(predicted_actions)
>>>>>>> Stashed changes

#                    if check_condition:
#                        if predicted_actions != previous_predicted_actions[-1][i]:
#                            print 'BEFORE:    ', previous_predicted_actions[-1][i]
#                            print 'THIS TIME: ', predicted_actions
#                            print 'TRUE:      ', u''.join(edit_actions) + STOP_CHAR
#                            print 'PRED:      ', prediction
#                            print 'WORD:      ', word
#                            print

                    if prediction == word:
                        train_correct += 1
            
#                previous_predicted_actions.append(pred_acts)
                train_accuracy = train_correct / sanity_set_size
                print '\t...finished in {:.3f} sec'.format(time.time() - then)

#                if train_accuracy > best_train_accuracy:
#                    best_train_accuracy = train_accuracy

                # get dev accuracy
                print 'evaluating on dev...'
                then = time.time()
                dev_correct = 0.
                dev_loss = 0.
                for lemma, word, feats in dev_data:

                    loss, prediction, predicted_actions = ti.transduce(lemma, feats)
                    if prediction == word:
                        dev_correct += 1
                        tick = 'V'
                    else:
                        tick = 'X'
                    #if check_condition:
                        #print 'TRUE:    ', word
                        #print 'PRED:    ', prediction
                        #print tick
                        #print
                    dev_loss += loss.scalar_value()
                dev_accuracy = dev_correct / dev_len
                print '\t...finished in {:.3f} sec'.format(time.time() - then)

                # save model
                if dev_accuracy > best_dev_accuracy:
                    best_train_accuracy = train_accuracy
                    best_dev_accuracy = dev_accuracy
                    # save best model to disk
                    model.save(tmp_model_path)
                    print 'saved new best transition model to {}'.format(tmp_model_path)
                    model_A.save(tmp_model_path_align)
                    print 'saved new best alignment model to {}'.format(tmp_model_path_align)
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

                print ('epoch: {0} Model B: train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} '
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

    # eval on dev #TODO - alignments should come from model A?
    print '=========DEV EVALUATION:========='
    model = dy.Model()
    if arguments['--second_hidden_layer']:
        ti = TwoLayerTransitionInflector(model, train_set, arguments)
    else:
        ti = TransitionInflector(model, train_set, arguments)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.populate(tmp_model_path)

    accuracy, dev_results = get_accuracy_predictions(ti, dev_set)
    print 'accuracy: {}'.format(accuracy)
    output_file_path = results_file_path + '.best'

    common.write_results_file_and_evaluate_externally(hyperparams, accuracy,
                                                      train_path, dev_path, output_file_path,
                                                      # nbest=True simply means inflection comes as a list
                                                      {i : v for i, v in enumerate(dev_results)},
                                                      nbest=True, test=False)
    if test_set:
        # eval on test
        print '=========TEST EVALUATION:========='
        accuracy, test_results = get_accuracy_predictions(ti, test_set)
        print 'accuracy: {}'.format(accuracy)
        output_file_path = results_file_path + '.best.test'

        common.write_results_file_and_evaluate_externally(hyperparams, accuracy,
                                                          train_path, test_path, output_file_path,
                                                          # nbest=True simply means inflection comes as a list
                                                          {i : v for i, v in enumerate(test_results)},
                                                          nbest=True, test=True)
