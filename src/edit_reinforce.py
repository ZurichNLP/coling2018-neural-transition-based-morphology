"""Trains and evaluates a state-transition model for inflection generation, using the sigmorphon 2017 shared task
data files and evaluation script. This model is trained with reinforcement learning.

Usage:
  transition_inflector_.py [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch ON]
  [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--action-input=ACTION] [--layers=LAYERS]
  [--dropout=DROPOUT] [--second_hidden_layer]
  [--optimization=OPTIMIZATION] [--epochs=EPOCHS] [--patience=PATIENCE] [--pretrain_copy=EPOCHS]
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
  --dynet-autobatch ON          perform autombatching
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
"""

from __future__ import division
from docopt import docopt
import sys
import os
import sys
import codecs
import random
import progressbar
import time
from collections import Counter, defaultdict

from defaults import DATA_PATH
import common
import editdistance

import dynet as dy
import numpy as np
from sklearn.preprocessing import scale

from stackLSTM_parser import Vocab, StackRNN
from transition_inflector import (TransitionInflector, log_to_file, get_accuracy_predictions,
                                  smart_align, dumb_align, ActionDataSet, MAX_ACTION_SEQ_LEN,
                                  OPTIMIZERS, COPY_CHAR, DELETE_CHAR, STOP_CHAR, UNK_CHAR,
                                  DeleteRNN, StackBiRNN)


class SamplableTransitionInflector(TransitionInflector):

    # Returns an expression of the loss for the sequence of actions.
    # actions can be oracle actions, actions predicted greedily, actions sampled according to softmaxes
    def transduce(self, lemma, feats, form=None, oracle_actions=None, sampling=True, external_cg=False):
        def _valid_actions(stack, buffer):
            valid_actions = []
            if len(buffer) > 0:
                valid_actions += [self.COPY, self.DELETE]
            else:
                valid_actions += [self.STOP]
            valid_actions += self.INSERTS
            return valid_actions

        # encode of oracle actions
        if oracle_actions:
            oracle_actions = [self.vocab_acts.w2i[a] for a in oracle_actions]
            oracle_actions += [self.STOP]
            oracle_actions = list(reversed(oracle_actions))

        # encoding of features: 1 if feature is on, 0 otherwise
        feats = [k + u'=' + v for k, v in feats.iteritems()]
        features = np.zeros(self.NUM_FEATS)
        for f in feats:
            f_id = self.vocab_feats.w2i.get(f, self.UNK_FEAT)
            features[f_id] = 55.
        #print features

        if not external_cg:
            dy.renew_cg()
        
        if form:
            form = list(form)
        features = dy.inputTensor(features)

        stack  = StackRNN(self.stackRNN, self.pempty_stack_emb)
        buffer = StackBiRNN(self.fbuffRNN, self.bbuffRNN, self.pempty_buffer_emb)
        delete = DeleteRNN(self.deleteRNN, self.pempty_delete_emb)  # has method to empty stack
        action_stack = StackRNN(self.actRNN, self.pempty_act_emb)

        W_s2h = dy.parameter(self.pW_s2h)   # state to hidden
        b_s2h = dy.parameter(self.pb_s2h)
        W_act = dy.parameter(self.pW_act)   # hidden to action
        b_act = dy.parameter(self.pb_act)

        # push the characters onto the buffer
        lemma_enc = []
        lemma_emb = []
        for char_ in reversed(lemma):
            char_id = self.vocab_chars.w2i.get(char_, self.UNK)
            char_embedding = self.CHAR_LOOKUP[char_id]
            lemma_enc.append((char_embedding, char_))
            lemma_emb.append(char_embedding)
        # char encodings and especially char themselves are put into storage
        # associated with each buffer state. On COPY action, we will simply
        # output character from storage without any decoding.
        buffer.transduce(lemma_emb, lemma_enc)

        losses = []
        word = []
        action_history = [None]
        while not ((len(buffer) == 0 and action_history[-1] == self.STOP) or len(action_history) == MAX_ACTION_SEQ_LEN):
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            valid_actions = _valid_actions(stack, buffer)
            p_t = dy.concatenate([buffer.embedding(), stack.embedding(),
                                  delete.embedding(), action_stack.embedding(), features])
            h = dy.rectify(W_s2h * p_t + b_s2h)
            #if self.dropout and oracle_actions:
            #    # apply inverted dropout at training
            #    dy.dropout(h, self.dropout)
            logits = W_act * h + b_act
            log_probs = dy.log_softmax(logits, valid_actions)
            if oracle_actions is None:
                if sampling:
                    dist = np.exp(log_probs.npvalue()) #**0.9
                    dist = dist / np.sum(dist)
                    # sample according to softmax
                    rand = np.random.rand()
                    for action, p in enumerate(dist):
                        rand -= p
                        if rand <= 0: break                    
                else:
                    action = np.argmax(log_probs.npvalue())
            else:
                action = oracle_actions.pop()
                
            losses.append(dy.pick(log_probs, action))
            action_history.append(action)
            action_stack.push(self.ACT_LOOKUP[action])

            #print 'Prob of COPY: ', np.exp(log_probs.npvalue())[self.COPY]
            
            # execute the action to update the transducer state
            #print action, self.vocab_acts.i2w[action]
            if action == self.COPY:
                char_embedding, char_ = buffer.pop()
                stack.push(char_embedding)
                word.append(char_)
            elif action == self.DELETE:
                char_embedding, char_ = buffer.pop()
                # deleted chars go onto delete stack
                delete.push(char_embedding)
            elif action == self.STOP:
                break
            else: # one of the INSERT actions
                insert_char = self.vocab_acts.i2w[action]
                # by construction, insert_char is always in vocab_chars
                insert_char_id = self.vocab_chars.w2i[insert_char]
                insert_char_embedding = self.CHAR_LOOKUP[insert_char_id]
                stack.push(insert_char_embedding)
                # we empty delete stack on INSERT
                delete.clear_all()
                word.append(insert_char)
                
            if form and word != form[:len(word)]: break

        word = u''.join(word)
        action_history = u''.join([self.vocab_acts.i2w[a] for a in action_history[1:]])
        return ((losses if losses else None), word, action_history)  #-dy.average(losses) ...


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
    train_data = ActionDataSet.from_file(train_path)
    dev_data = ActionDataSet.from_file(dev_path)
    if test_path:
        test_data = ActionDataSet.from_file(test_path)
    else:
        test_data = None

    print 'Checking if any special symbols in data...'
    for data, name in [(train_data, 'train'), (dev_data, 'dev')] + \
        ([(test_data, 'test')] if test_data else []):
        data = set(data.lemmas + data.words)  # for test words are 'COVER' ..?
        for c in [COPY_CHAR, DELETE_CHAR, STOP_CHAR, UNK_CHAR]:
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
    if arguments['--second_hidden_layer']:
        sys.exit(1)
    else:
        ti = SamplableTransitionInflector(model, train_data, arguments)

    optimization = arguments['--optimization']
    epochs = int(arguments['--epochs'])
    max_patience = int(arguments['--patience'])
    pretrain_epochs = int(arguments['--pretrain_copy'])

    hyperparams = {'ALIGNER': 'DUMB' if is_dumb else 'SMART',
                   'ALIGN_ITERATIONS': align_iterations if not is_dumb else 'does not apply',
                   'TRY_REVERSE': try_reverse if is_dumb else 'does not apply',
                   'MAX_ACTION_SEQ_LEN': MAX_ACTION_SEQ_LEN,
                   'OPTIMIZATION': optimization,
                   'EPOCHS': epochs,
                   'PATIENCE': max_patience,
                   'BEAM_WIDTH': 1,
                   'FEATURE_TYPE_DETECT': None,
                   'PRETRAINING EPOCHS': pretrain_epochs}
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

        batch_size = 5
        sample_size = 20
        
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
            ti.set_dropout()
            # sample a minibatch, produce reward and update
            train_set = list(train_data.iter(shuffle=False)) # True
            batches = (train_set[q:q+batch_size] for q in range(1, train_len, batch_size))
            i = 0
            for batch_no, batch in enumerate(batches):
                # here we do training with sampling and reinforcement learning
                dy.renew_cg()
                #batch_loss = dy.scalarInput(0.)
                batch_loss = []
                rewards = []
                for (lemma, word, actions, feats) in batch:
                    if lemma == word:
                        print 'Dropping ', lemma, feats
                        continue
                    #lemma_len = len(lemma)
                    #lemma_len_inv = 1. / lemma_len
                    for w in xrange(sample_size):
                        #if w + 1 == sample_size:
                            # adding supervised signal
                            #oracle_actions = actions
                        #else:
                            #oracle_actions = None
                        loss, prediction, predicted_actions = ti.transduce(lemma, feats, form=word,
                                                                            #oracle_actions=oracle_actions,
                                                                            sampling=True, external_cg=True)
                        reward = 2. if prediction == word else -1                                # reward for accuracy
                        if prediction == word:
                            print lemma, word, prediction, predicted_actions
                        #reward += sum(c == COPY_CHAR for c in prediction_actions)                   # reward for copying
                        #reward += - 0.1 * (len(prediction_actions) - lemma_len)                     # reward for length
                        #reward += -int(editdistance.eval(word, prediction))  # the smaller the distance the better
                        reward += 0.1 * (len(prediction) - 1) if prediction != word else len(prediction)
                        #print prediction, word, reward
                        #batch_loss += dy.scalarInput(reward) * loss
                        batch_loss.append(-dy.average(loss))
                        rewards.append(reward)
                    i += 1
                    if i < 5:
                        print prediction, word, predicted_actions, reward, [float(np.exp(l.npvalue())) for l in loss]
                #rewards = scale(rewards) #+ 1.
                #print rewards
                batch_loss = dy.dot_product(dy.inputVector(rewards), dy.concatenate(batch_loss))
                total_loss += batch_loss.scalar_value()
                print 'Batch loss, batch reward: ', batch_loss.npvalue(), sum(rewards) / (sample_size * batch_size)
                batch_loss.backward()
                trainer.update()
                avg_loss = total_loss / (i + epoch * train_len)
            # DISABLE DROPOUT AFTER TRAINING
            ti.disable_dropout()
            print '\t...finished in {:.3f} sec'.format(time.time() - then)
            print 'Total loss: ', total_loss
            # condition for displaying stuff like true words and predictions
            check_condition = (epoch > 0 and (epoch % 5 == 0 or epoch == epochs - 1))

            # get train accuracy
            print 'evaluating on train...'
            then = time.time()
            train_correct = 0.
            pred_acts = []
            for i, (lemma, word, actions, feats) in enumerate(train_data.iter(indices=sanity_set_size)):
                _, prediction, predicted_actions = ti.transduce(lemma, feats)
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

            if epoch < 7:
                print 'Skipping evaluation on dev...'
                
                print ('epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} '
                       'train accuracy: {4:.4f} best dev accuracy: {5:.4f} best train accuracy: {6:.4f} '
                       'patience = {7}').format(epoch, avg_loss, 0, 0, train_accuracy, 0, best_train_accuracy, patience)
                continue

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
                if check_condition:
                    print 'TRUE:    ', word
                    print 'PRED:    ', prediction
                    print tick
                    print
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
    if arguments['--second_hidden_layer']:
        sys.exit(1)
    else:
        ti = SamplableTransitionInflector(model, train_data, arguments)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)

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
