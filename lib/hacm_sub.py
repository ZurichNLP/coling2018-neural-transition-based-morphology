from __future__ import division

import dynet as dy
import numpy as np

import datasets
from defaults import (STEP, BEGIN_WORD, END_WORD, UNK, UNK_CHAR, MAX_ACTION_SEQ_LEN)
from stack_lstms import Encoder
from transducer import Transducer

class MinimalTransducer(Transducer):
    
    def __init__(self, model, vocab, **kwargs):
        super(MinimalTransducer, self).__init__(model, vocab, **kwargs)
        self.subst_i2w = {i : w[0] for w, i in self.vocab.act.w2i.iteritems()
                          if i < self.NUM_ACTS and len(w) == 2 and w.endswith(u'@')}
        self.SUBSTS = self.subst_i2w.keys()
        print '{} SUBSTITUTION ACTIONS: '.format(len(self.SUBSTS))
        print u', '.join(self.vocab.act.i2w[a] for a in self.SUBSTS)
        print 'SUBST dictionary:'
        print u', '.join(u'({}={})'.format(k, v) for k, v in self.subst_i2w.items())
        self.INSERTS = [a for a in self.INSERTS if a not in self.SUBSTS]
    
    def _classifier(self, model):
        # Here, we add a copy logistic classifier.
        # input + decoder output for good measure
        gen_dim = self.WORD_REPR_DIM + self.CLASSIFIER_IMPUT_DIM
        self.pW_gen = model.add_parameters((1, gen_dim))
        self.pb_gen = model.add_parameters(1)
        print ' * COPY LOGISTIC:     IN-DIM: {}, OUT-DIM: {}'.format(gen_dim, 1)
        super(MinimalTransducer, self)._classifier(model)
    
    def transduce(self, lemma, feats, oracle_actions=None, external_cg=True, sampling=False, unk_avg=True,
                  debug_mode=True):
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
#        def _valid_actions(encoder):
#            valid_actions = []
#            if len(encoder) > 0:
#                valid_actions += [STEP]
#            else:
#                valid_actions += [END_WORD]
#            valid_actions += self.INSERTS
#            return valid_actions

        if not external_cg:
            dy.renew_cg()

        debug_mode = False
        
        #print 'First oracle action: ', oracle_actions[0], self.vocab.act.i2w[oracle_actions[0]] 
            
        if oracle_actions:
            # reverse to enable simple popping
            oracle_actions = oracle_actions[::-1]
            oracle_actions.pop()  # COPY of BEGIN_WORD_CHA           
            # vectorize lemma with UNK

        # vectorize lemma
        lemma_enc = self._build_lemma(lemma, unk_avg, is_training=bool(oracle_actions))
            
        # vectorize features
        features = self._build_features(*feats)
            
        # add encoder and decoder to computation graph
        encoder = Encoder(self.fbuffRNN, self.bbuffRNN)
        decoder = self.wordRNN.initial_state()
        
        # encoder is a stack which pops lemma characters and their
        # representations from the top.
        encoder.transduce(lemma_enc, lemma)

        # add classifier to computation graph
        if self.MLP_DIM:
            # decoder output to hidden
            W_s2h = dy.parameter(self.pW_s2h)
            b_s2h = dy.parameter(self.pb_s2h)
        # hidden to action
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)
        # decoder output to copy
        W_gen = dy.parameter(self.pW_gen)
        b_gen = dy.parameter(self.pb_gen)
        
        char_enc = encoder.pop()  # BEGIN_WORD_CHAR
        action_history = [char_enc]  # doesn't matter... this is wrong btw
        word = []
        losses = []
        count = 0
        
        if debug_mode:
            print
            if oracle_actions: print u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
            print u''.join([self.vocab.char.i2w[a] for a in lemma])
        
        while len(action_history) <= MAX_ACTION_SEQ_LEN:
            
            encoder_embedding, char_enc = encoder.embedding(extra=True)

            if debug_mode:
                print 'Previous action: ', action_history[-1], self.vocab.act.i2w[action_history[-1]]
                print 'Encoder last ind, char: ', lemma, char_enc, self.vocab.char.i2w[char_enc]
                print 'word: ', u''.join(word)
#                print 'Remaining actions: ', oracle_actions, u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
#                count += 1
            #elif action_history[-1] >= self.NUM_ACTS:
            #    print 'Will be adding unseen act embedding: ', self.vocab.act.i2w[action_history[-1]]
            
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            
            # decoder
            decoder_input = dy.concatenate([encoder_embedding,
                                            features,
                                            self.ACT_LOOKUP[action_history[-1]]
                                           ])
            decoder = decoder.add_input(decoder_input)
            decoder_output = decoder.output()

            # generate
            if self.MLP_DIM:
                h = self.NONLIN(W_s2h * decoder_output + b_s2h)
            else:
                h = decoder_output

            logits = W_act * h + b_act
            probs_gen = dy.softmax(logits)
#            valid_actions = np.ones(self.NUM_ACTS) * -np.inf
#            valid_actions[_valid_actions(encoder)] = 0.
#            valid_actions = dy.inputTensor(valid_actions)
#            probs_gen = dy.softmax(logits + valid_actions)
             # log_probs = dy.log_softmax(logits, valid_actions)

            if char_enc >= self.NUM_CHARS: #unk char in lemma
                # copy unk
                encoder.pop()
                action = STEP
                losses.append(-dy.log(dy.pick(probs_gen, action)))
                action_history.append(action)
                char_ = self.vocab.char.i2w[char_enc]
                word.append(char_)
                if debug_mode: print 'Uknown lemma char: ', char_enc, self.vocab.char.i2w[char_enc]
            else:
                if char_enc != END_WORD:
#                    copy_action = self.vocab.act.w2i[self.vocab.char.i2w[char_enc] + u'@']
                    copy_action = self.vocab.act.w2i[self.vocab.char.i2w[char_enc]+u'@'
                                 if self.vocab.char.i2w[char_enc]+u'@' in self.vocab.act else UNK_CHAR]
                else:
                    copy_action = END_WORD
                if debug_mode: print 'Copy action ind, char: ', copy_action, self.vocab.act.i2w[copy_action]
                probs_copy = np.zeros(self.NUM_ACTS)
#                if copy_action in _valid_actions(encoder):
#                    probs_copy[copy_action] = 1.
                probs_copy[copy_action] = 1.
                probs_copy = dy.inputTensor(probs_copy)
                # copy switch
                gen = dy.logistic(W_gen * dy.concatenate([decoder_output, decoder_input]) + b_gen)
#                probs = (probs_gen * (1 - gen)) + (probs_copy * (gen))
                probs = dy.cmult(gen, probs_gen) + dy.cmult(1-gen, probs_copy)
                # get action (argmax, sampling, or use oracle actions)
                
    #            print gen.scalar_value(), probs_copy.npvalue()
                    #, probs_gen.npvalue(), probs.npvalue()
                if oracle_actions is None:
                    if sampling:
                        dist = probs.npvalue() #**0.9
                        # sample according to softmax
                        rand = np.random.rand()
                        for action, p in enumerate(dist):
                            rand -= p
                            if rand <= 0: break
                    else:
                        action = np.argmax(probs.npvalue())
                else:
                    action = oracle_actions.pop()

                losses.append(dy.log(dy.pick(probs, action)))
                action_history.append(action)
                # execute the action to update the transducer state                
                if action in self.SUBSTS:
                    # 1. Append new character to the output word
                    char_ = self.subst_i2w[action]
                    word.append(char_)
                    # 2. Pop encoded character
                    if char_enc != END_WORD:
                        encoder.pop()
                    else: pass
                        #print 'Cannot pop!'
                        #print 'Previous action: ', action_history[-1], self.vocab.act.i2w[action_history[-1]]
                        #print 'Encoder last ind, char: ', lemma, char_enc, self.vocab.char.i2w[char_enc]
                        #print 'word: ', u''.join(word)                        
                elif action == STEP:
                    # 1. Pop encoded character
                    if char_enc != END_WORD:
                        encoder.pop()
                elif action == END_WORD:
                    # 1. Finish transduction
                    break
                else:
                    # one of the INSERT actions
                    assert action in self.INSERTS, (
                            u'action: {} ({}), SUBSTS: {}, INSERTS: {}'.format(
                            action, self.vocab.act.i2w[action], self.SUBSTS, self.INSERTS))
                    # 1. Append inserted character to the output word
                    char_ = self.vocab.act.i2w[action]
                    word.append(char_)
                
        word = u''.join(word)
        return losses, word, action_history
