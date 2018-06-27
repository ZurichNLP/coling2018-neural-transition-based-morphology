"""Trains and evaluates a state-transition model for inflection generation, using the sigmorphon 2017 shared task
data files and evaluation script.

Usage:
  transition_inflector_.py [--dynet-seed SEED] [--dynet-mem MEM]
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

#sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
#sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
#sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


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


def smart_align(pairs, align_symbol=ALIGN_SYMBOL,
                iterations=150, burnin=5, lag=1, mode='crp'):
    return align.Aligner(pairs,
                         align_symbol=align_symbol,
                         iterations=iterations,
                         burnin=burnin,
                         lag=lag,
                         mode=mode).alignedpairs


def dumb_align(pairs, align_symbol=ALIGN_SYMBOL, **kwargs):
    return common.dumb_align(pairs, align_symbol)


class ActionDataSet(HardDataSet):

    def __init__(self, *args, **kwargs):

        super(ActionDataSet, self).__init__(*args)

        self.aligner = None
        self.oracle_actions = None
        self.action_set = None

    def align(self, aligner=smart_align, **kwargs):
        print 'started aligning'
        pairs = zip(self.lemmas, self.words)
        self.aligner = aligner
        print 'aligning with %s' % self.aligner
        aligned_pairs = aligner(pairs, **kwargs)
        self.set_aligned(aligned_pairs)
        print 'finished aligning'

    def iter(self, indices=None, shuffle=False):
        # quick fix for test set
        oracle_actions = self.oracle_actions if self.oracle_actions else [None]*self.length
        zipped = zip(self.lemmas, self.words, oracle_actions, self.feat_dicts)
        if indices or shuffle:
            if not indices:
                indices = range(self.length)
            elif isinstance(indices, int):
                indices = range(indices)
            else:
                assert isinstance(indices, (list, tuple))
            if shuffle:
                random.shuffle(indices)
            zipped = [zipped[i] for i in indices]
        return zipped

    def build_code(self, lemma, word):
        code = []
        true_code = []
        delete_stack = []
        inserts = []
        for l, w in zip(lemma, word):
            if l == w:
                # action COPY
                if delete_stack:
                    code.extend(inserts)
                    inserts = []
                    delete_stack = []
                code.append(COPY_CHAR)
                true_code.append('=')
            elif l == ALIGN_SYMBOL:
                # action INSERT w
                if delete_stack:
                    code.extend(inserts)
                    inserts = []
                code.append(w)
                true_code.append('+')
            elif w == ALIGN_SYMBOL:
                # action DELETE (put one delete stack)
                code.append(DELETE_CHAR)
                delete_stack.append(l)
                true_code.append('-')
            else:
                # action DELETE (put one delete stack)
                # action INSERT w
                code.append(DELETE_CHAR)
                delete_stack.append(l)
                inserts.append(w)
                true_code.append('?')
        code.extend(inserts)
        return code, true_code

    def build_edit_actions(self, code):
        actions = []
        for i, c in enumerate(code):
            if i == 0 or c == COPY_CHAR:
                actions.append(c)
                # count deletes and store inserts
                # between two copy actions
                inserts = []
                deletes = 0
                for b in code[i+1:]:
                    if b == COPY_CHAR:
                        # copy
                        break
                    elif b == DELETE_CHAR:
                        # delete
                        deletes += 1
                    else:
                        inserts.append(b)
                between_copies = [DELETE_CHAR]*deletes + inserts
                actions.extend(between_copies)

        return actions


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
                rcode, rtrue_code = self.build_code(new_al_lemma[::-1], new_al_word[::-1])
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
        self.action_set = sorted(self.action_set)
        print 'finished building oracle actions'
        print 'number of actions: {}'.format(len(self.action_set))

    def align_and_build_actions(self,
                                aligner=smart_align,
                                try_reverse=False,
                                verbose=False, **kwargs):
        self.align(aligner, **kwargs)
        self.build_oracle_actions(try_reverse=try_reverse, verbose=verbose)
        action_counter = Counter([a for acts in self.oracle_actions for a in acts])
        freq_DELETE = action_counter[DELETE_CHAR] / sum(action_counter.values())
        freq_COPY = action_counter[COPY_CHAR] / sum(action_counter.values())
        print ('Alignment results in: COPY action freq {:.3f}, '
               'DELETE action freq {:.3f}'.format(freq_COPY, freq_DELETE))
        if freq_COPY < 0.1:
            print 'WARNING: Too few COPY actions!\n'
        if freq_DELETE > 0.3:
            print 'WARNING: Many DELETE actions!\n'
        if verbose:
            print 'Examples of oracle actions:'
            for a in self.oracle_actions[:20]:
                print u''.join(a)


# infer lemma features from feat_dicts
def infer_lemma_feats(train_data):
    lemma_feats = defaultdict(list)
    pos_list = []
    for lemma, word, _, feat_dict in train_data.iter():
        pos = feat_dict['POS']
        if lemma == word:
            lemma_feats[pos].append(feat_dict)
        else:
            lemma_feats[pos]
        pos_list.append(pos)            
    lemma_feats = dict(lemma_feats)
    output = {}
    for pos, v in lemma_feats.items():
        v = [tuple(sorted(f.items())) for f in v]
        if v:
            c = Counter(v)
            [(_, count)] = c.most_common(1)
            fs = [dict(k) for k, v in c.items() if v == count]
        else:
            fs = [{'POS' : pos}]
        output[pos] = fs[0]  # we just pick the first...
    print 'Using infered features: '
    for k, v in output.items():
        print k, '\t', v
    return [output[pos] for pos in pos_list]


class DeleteRNN(StackRNN):
    def clear_all(self):
        self.s = self.s[:1]


class StackBiRNN(object):
    def __init__(self, frnn, brnn, p_empty_embedding = None):
        self.frnn = frnn
        self.brnn = brnn
        self.empty = None
        if p_empty_embedding:
            self.empty = dy.parameter(p_empty_embedding)
    def transduce(self, embs, extras=None):
        fs = self.frnn.initial_state()
        bs = self.brnn.initial_state()
        fs_states = fs.add_inputs(embs)
        bs_states = bs.add_inputs(reversed(embs))
        self.s = [(fs, bs, None)] + zip(fs_states, bs_states, extras)
    def pop(self):
        return self.s.pop()[-1] # return "extra" (i.e., whatever the caller wants or None)
    def embedding(self):
        if len(self.s) > 1:
            fs = self.s[-1][0].output()
            bs = self.s[-1][1].output()
            emb = dy.concatenate([fs, bs])
        else:
            # work around since inital_state.output() is None
            emb = self.empty
        return emb
    def __len__(self):
        return len(self.s) - 1


class TransitionInflector(object):
    def __init__(self, model, train_data, arguments):

        self.INPUT_DIM    = int(arguments['--input'])
        self.HIDDEN_DIM   = int(arguments['--hidden'])
        #self.feat_input_dim = int(arguments['--feat-input'])
        self.ACTION_DIM   = int(arguments['--action-input'])
        self.LAYERS       = int(arguments['--layers'])
        self.dropout      = float(arguments['--dropout'])

        self.build_vocabularies(train_data)
        self.build_model(model)
        # for printing
        self.hyperparams = {'INPUT_DIM'       : self.INPUT_DIM,
                            'HIDDEN_DIM'      : self.HIDDEN_DIM,
                            'FEAT_INPUT_DIM'  : None,
                            'ACTION_INPUT_DIM': self.ACTION_DIM,
                            'LAYERS'          : self.LAYERS,
                            'DROPOUT'         : self.dropout}

    def build_vocabularies(self, train_data):
        
        if not isinstance(train_data, ActionDataSet):
            # then it's a list of compatible w2i's
            self.vocab_acts  = Vocab(train_data['acts_w2i'])
            self.vocab_feats = Vocab(train_data['feats_w2i'])
            self.vocab_chars = Vocab(train_data['chars_w2i'])
        else:
            acts = train_data.action_set + [STOP_CHAR]
            self.vocab_acts = Vocab.from_list(acts)
            
            feats = set([k + u'=' + v for d in train_data.feat_dicts
                         for k, v in d.iteritems()] + [UNK_FEAT_CHAR])
            self.vocab_feats = Vocab.from_list(feats)            
            
            chars = set(u''.join(train_data.words + train_data.lemmas  + [UNK_CHAR]))
            self.vocab_chars = Vocab.from_list(chars)
        
        # ACTION VOCABULARY            
        self.COPY   = self.vocab_acts.w2i[COPY_CHAR]
        self.DELETE = self.vocab_acts.w2i[DELETE_CHAR]
        self.STOP   = self.vocab_acts.w2i[STOP_CHAR]
        # rest are INSERT_* actions
        INSERT_CHARS, INSERTS = zip(*[(a, a_id) for a, a_id in self.vocab_acts.w2i.iteritems()
                                      if a not in set([COPY_CHAR, DELETE_CHAR, STOP_CHAR])])
        self.INSERT_CHARS, self.INSERTS = list(INSERT_CHARS), list(INSERTS)
        self.NUM_ACTS = self.vocab_acts.size()
        print u'{} actions of which {} are INSERT actions: {}'.format(self.NUM_ACTS,
                                                                      len(self.INSERTS),
                                                                      u', '.join(self.INSERT_CHARS))
        # FEATURE VOCABULARY
        self.UNK_FEAT      = self.vocab_feats.w2i[UNK_FEAT_CHAR]
        self.FEATURE_TYPES = list(self.vocab_feats.w2i)
        self.NUM_FEATS     = self.vocab_feats.size()
        print '{} features. Feature types: {}'.format(self.NUM_FEATS,
                                                      u', '.join(self.FEATURE_TYPES))
        # CHARACTER VOCABULARY
        self.UNK       = self.vocab_chars.w2i[UNK_CHAR]
        self.NUM_CHARS = self.vocab_chars.size()
        print '{} characters'.format(self.NUM_CHARS)


    def build_model(self, model):

        # LSTMs for storing lemma, word, actions, and deleted characters
        # parameters: layers, in-dim, out-dim, model
        # BiLSTM for lemma
        self.fbuffRNN  = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)
        self.bbuffRNN  = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)
        # LSTM for word
        self.stackRNN  = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)
        # LSTM for actions: !! out-dim != hidden-dim, intended..?
        self.actRNN    = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.ACTION_DIM, model)
        # LSTM for deleted characters
        self.deleteRNN = dy.LSTMBuilder(self.LAYERS, self.INPUT_DIM, self.HIDDEN_DIM, model)

        # empty embeddings for all LSTM above
        self.pempty_buffer_emb = model.add_parameters(2*self.HIDDEN_DIM)
        self.pempty_stack_emb  = model.add_parameters(self.HIDDEN_DIM)
        self.pempty_act_emb    = model.add_parameters(self.ACTION_DIM)  # !!
        self.pempty_delete_emb = model.add_parameters(self.HIDDEN_DIM)

        # embedding lookups for characters and actions
        self.CHAR_LOOKUP = model.add_lookup_parameters((self.NUM_CHARS, self.INPUT_DIM))
        self.ACT_LOOKUP  = model.add_lookup_parameters((self.NUM_ACTS, self.INPUT_DIM))
        #self.FEAT_LOOKUP = model.add_lookup_parameters((self.NUM_FEATS, self.FEAT_INPUT_DIM))

        # transducer state to hidden
        # FEATURE VECTOR + 5 LSTMs: 2x lemma, 1x word, 1x delete, and 1x actions
        in_dim = self.HIDDEN_DIM*4 + self.ACTION_DIM + self.NUM_FEATS
        self.pW_s2h = model.add_parameters((self.HIDDEN_DIM, in_dim))
        self.pb_s2h = model.add_parameters(self.HIDDEN_DIM)

        # hidden to action
        self.pW_act = model.add_parameters((self.NUM_ACTS, self.HIDDEN_DIM))
        self.pb_act = model.add_parameters(self.NUM_ACTS)

        print 'Model dimensions:'
        print ' * LEMMA biLSTM (aka BUFFER): IN-DIM: {}, OUT-DIM: {}'.format(2*self.INPUT_DIM,
                                                                             2*self.HIDDEN_DIM)
        print ' * WORD LSTM (aka STACK):     IN-DIM: {}, OUT-DIM: {}'.format(self.INPUT_DIM, self.HIDDEN_DIM)
        print ' * ACTION LSTM:               IN-DIM: {}, OUT-DIM: {}'.format(self.INPUT_DIM, self.ACTION_DIM)
        print ' * DELETE LSTM:               IN-DIM: {}, OUT-DIM: {}'.format(self.INPUT_DIM, self.HIDDEN_DIM)
        print ' All LSTMs have {} layer(s)'.format(self.LAYERS)
        print
        print ' * CHAR EMBEDDING LAYER:      IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_CHARS, self.INPUT_DIM)
        print ' * ACTION EMBEDDING LAYER:    IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_ACTS, self.INPUT_DIM)
        print
        print ' * TRANSDUCER STATE 2 HIDDEN: IN-DIM: {}, OUT-DIM: {}'.format(in_dim, self.HIDDEN_DIM)
        print ' * SOFTMAX:                   IN-DIM: {}, OUT-DIM: {}'.format(self.HIDDEN_DIM, self.NUM_ACTS)
        #print

        if self.dropout:
            def set_dropout():
                self.stackRNN.set_dropout(self.dropout)
                self.deleteRNN.set_dropout(self.dropout)
                #self.actRNN.set_dropout(self.dropout)

            def disable_dropout():
                self.stackRNN.disable_dropout()
                self.deleteRNN.disable_dropout()
                #self.actRNN.disable_dropout()

            self.set_dropout = set_dropout
            self.disable_dropout = disable_dropout
            print 'Using dropout of {} on STACK and DELETE LSTMs'.format(self.dropout)
        else:
            self.set_dropout = lambda: None
            self.disable_dropout = lambda: None
            print 'Not using dropout'

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def transduce(self, lemma, feats, oracle_actions=None, external_cg=False):
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
                action = np.argmax(log_probs.npvalue())
            else:
                action = oracle_actions.pop()

            losses.append(dy.pick(log_probs, action))
            action_history.append(action)
            action_stack.push(self.ACT_LOOKUP[action])

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

        word = u''.join(word)
        action_history = u''.join([self.vocab_acts.i2w[a] for a in action_history[1:]])
        return ((-dy.average(losses) if losses else None), word, action_history)


class TwoLayerTransitionInflector(TransitionInflector):
    def __init__(self, model, train_data, arguments):
        
        super(TwoLayerTransitionInflector, self).__init__(model, train_data, arguments)

        self.build_model(model)
        self.hyperparams['2ND_HIDDEN_LAYER'] = True

    def build_model(self, model):
        
        super(TwoLayerTransitionInflector, self).build_model(model)       

        # second layer: hidden to hidden
        self.pW_s2h2 = model.add_parameters((self.HIDDEN_DIM, self.HIDDEN_DIM))
        self.pb_s2h2 = model.add_parameters(self.HIDDEN_DIM)
        print ' * 2ND TRANS. STATE 2 HIDDEN: IN-DIM: {}, OUT-DIM: {}'.format(self.HIDDEN_DIM, self.HIDDEN_DIM)
        if self.dropout:
            print 'Using dropout of {} on 1ST TRANS. STATE TO HIDDEN LAYER'.format(self.dropout)

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def transduce(self, lemma, feats, oracle_actions=None):
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

        dy.renew_cg()

        features = dy.inputTensor(features)

        stack  = StackRNN(self.stackRNN, self.pempty_stack_emb)
        buffer = StackBiRNN(self.fbuffRNN, self.bbuffRNN, self.pempty_buffer_emb)
        delete = DeleteRNN(self.deleteRNN, self.pempty_delete_emb)  # has method to empty stack
        action_stack = StackRNN(self.actRNN, self.pempty_act_emb)

        W_s2h = dy.parameter(self.pW_s2h)   # state to hidden
        b_s2h = dy.parameter(self.pb_s2h)
        W_act = dy.parameter(self.pW_act)   # hidden to action
        b_act = dy.parameter(self.pb_act)
        
        W_s2h2 = dy.parameter(self.pW_s2h2) # 2nd layer: state to hidden
        b_s2h2 = dy.parameter(self.pb_s2h2)

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
            h1 = dy.rectify(W_s2h * p_t + b_s2h)
            if self.dropout and oracle_actions:
                # apply inverted dropout at training
                dy.dropout(h1, self.dropout)
            h = dy.rectify(W_s2h2 * h1 + b_s2h2)
            logits = W_act * h + b_act
            log_probs = dy.log_softmax(logits, valid_actions)
            if oracle_actions is None:
                action = np.argmax(log_probs.npvalue())
            else:
                action = oracle_actions.pop()

            losses.append(dy.pick(log_probs, action))
            action_history.append(action)
            action_stack.push(self.ACT_LOOKUP[action])

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

        word = u''.join(word)
        action_history = u''.join([self.vocab_acts.i2w[a] for a in action_history[1:]])
        return ((-dy.average(losses) if losses else None), word, action_history)


def log_to_file(log_file_name, e, avg_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'avg_loss', 'train_accuracy', 'dev_accuracy')

    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_loss, train_accuracy, dev_accuracy))


def get_accuracy_predictions(ti, test_data):
    correct = 0.
    final_results = []
    for lemma, word, actions, feats in test_data.iter():
        loss, prediction, predicted_actions = ti.transduce(lemma, feats)
        if prediction == word:
            correct += 1
        final_results.append((lemma, feats, [prediction]))  # pred expected as list
    accuracy = correct / test_data.length
    return accuracy, final_results


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
        ti = TwoLayerTransitionInflector(model, train_data, arguments)
    else:
        ti = TransitionInflector(model, train_data, arguments)

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

        if pretrain_epochs:
            print 'Pretraining using copy data ...'
            copy_data = ActionDataSet(train_data.lemmas, train_data.lemmas, infer_lemma_feats(train_data))
            copy_data.oracle_actions = [[COPY_CHAR]*len(l) for l in train_data.lemmas]

            for epoch in range(pretrain_epochs):
                copy_loss = 0.0
                ti.set_dropout()
                for lemma, word, actions, feats in copy_data.iter(shuffle=True):
                    # here we do training
                    loss, _, _ = ti.transduce(lemma, feats, actions)  # STOP
                    if loss is not None:
                        copy_loss += loss.scalar_value()
                        loss.backward()
                        trainer.update()
                ti.disable_dropout()
                # predict train set and dev set
                copy_correct = 0.
                for j, (lemma, word, actions, feats) in enumerate(copy_data.iter(indices=100)):
                    _, prediction, predicted_actions = ti.transduce(lemma, feats)
                    if prediction == word:
                        copy_correct += 1
                    else:
                        print 'TRUE:    ', word
                        print 'PRED:    ', prediction
                        print 'X'
                copy_accuracy = copy_correct / 100
                print 'Pretraining: epoch {}, train accuracy {}, train loss {}'.format(epoch,
                                                                                       copy_accuracy,
                                                                                       copy_loss)
            print 'Finished pretraining...'

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
            ti.enable_dropout()
            # compute loss for each sample and update
            for i, (lemma, word, actions, feats) in enumerate(train_data.iter(shuffle=True)):
                # here we do training
                loss, _, _ = ti.transduce(lemma, feats, actions)
                if loss is not None:
                    total_loss += loss.scalar_value()
                    loss.backward()
                    trainer.update()
                if i > 0:
                    avg_loss = total_loss / (i + epoch * train_len)
                else:
                    avg_loss = total_loss
            # DISABLE DROPOUT AFTER TRAINING
            ti.disable_dropout()
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
        ti = TwoLayerTransitionInflector(model, train_data, arguments)
    else:
        ti = TransitionInflector(model, train_data, arguments)
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
