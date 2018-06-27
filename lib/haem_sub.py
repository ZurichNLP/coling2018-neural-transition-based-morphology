from __future__ import division

import dynet as dy
import numpy as np

import datasets
from defaults import (COPY, DELETE, BEGIN_WORD, END_WORD, UNK, MAX_ACTION_SEQ_LEN)
from stack_lstms import Encoder, StackRNN, StackBiRNN
from transducer import Transducer

class EditTransducer(Transducer):

    def __init__(self, model, vocab, **kwargs):
        
        super(EditTransducer, self).__init__(model, vocab, **kwargs)
        
        self.subst_i2w = {i : w[0] for w, i in self.vocab.act.w2i.iteritems()
                          if i < self.NUM_ACTS and len(w) == 2 and w.endswith(u'@')}
        self.SUBSTS = self.subst_i2w.keys()
        print '{} SUBSTITUTION ACTIONS: '.format(len(self.SUBSTS))
        print u', '.join(self.vocab.act.i2w[a] for a in self.SUBSTS)
        print 'SUBST dictionary:'
        print u', '.join(u'({}={})'.format(k, v) for k, v in self.subst_i2w.items())
        self.INSERTS = [a for a in self.INSERTS if a not in self.SUBSTS]
    
    def transduce(self, lemma, feats, oracle_actions=None, external_cg=True, sampling=False, unk_avg=True):
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
        def _valid_actions(encoder):
            valid_actions = []
            if len(encoder) > 1:
                valid_actions += [COPY, DELETE] + self.SUBSTS
            else:
                valid_actions += [END_WORD]
            valid_actions += self.INSERTS
            return valid_actions
        
        show_oracle_actions = False #not oracle_actions

        if not external_cg:
            dy.renew_cg()

        if oracle_actions:
            # reverse to enable simple popping
            oracle_actions = oracle_actions[::-1]
            oracle_actions.pop()  # COPY of BEGIN_WORD_CHAR

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
    
        char_ = encoder.pop()  # BEGIN_WORD_CHAR
        action_history = [char_]  # COPY
        word = []
        losses = []
        count = 0
        
        if show_oracle_actions and oracle_actions:
            print
            print u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
            print u''.join([self.vocab.char.i2w[a] for a in lemma])
        
        while len(action_history) <= MAX_ACTION_SEQ_LEN:
            
            if show_oracle_actions:
                print 'Action: ', count, self.vocab.act.i2w[action_history[-1]]
                print 'Encoder length, char: ', lemma, len(encoder), self.vocab.char.i2w[encoder.s[-1][-1]]
                print 'word: ', u''.join(word)
                if oracle_actions:
                    print 'Remaining actions: ', oracle_actions, u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
                count += 1
            #elif action_history[-1] >= self.ACT_TRAIN:
            #    print 'Will be adding unseen act embedding: ', self.vocab.act.i2w[action_history[-1]]
            
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            valid_actions = _valid_actions(encoder)
            
            # decoder
            decoder_input = dy.concatenate([encoder.embedding(),
                                            features,
                                            self.ACT_LOOKUP[action_history[-1]]
                                           ])
            decoder = decoder.add_input(decoder_input)
            
            # classifier
            if self.double_feats:
                classifier_input = dy.concatenate([decoder.output(), features])
            else:
                classifier_input = decoder.output()
            if self.MLP_DIM:
                h = self.NONLIN(W_s2h * classifier_input + b_s2h)
            else:
                h = classifier_input
            logits = W_act * h + b_act
            log_probs = dy.log_softmax(logits, valid_actions)
            # get action (argmax, sampling, or use oracle actions)
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

            assert action in _valid_actions(encoder), (u'{} ({}), loss: {} , log_probs: {}'.format(action,
                self.vocab.act.i2w[action], losses[-1].scalar_value(), log_probs.npvalue()).encode('utf8'))
                
            #print 'action, log_probs: ', action, self.vocab.act.i2w[action], losses[-1].scalar_value(), log_probs.npvalue()
            
            # execute the action to update the transducer state
            if action == COPY:
                # 1. Increment attention index
                char_ = encoder.pop()
                # 2. Append copied character to the output word
                word.append(self.vocab.char.i2w[char_])
            elif action == DELETE:               
                # 1. Increment attention index
                encoder.pop()
            elif action == END_WORD:
                # 1. Finish transduction
                break
            elif action in self.SUBSTS:
                # 1. Append new character to the output word
                char_ = self.subst_i2w[action]
                word.append(char_)
                # 2. Pop replaced character
                encoder.pop()
            else:
                # one of the INSERT actions
                assert action in self.INSERTS
                # 1. Append inserted character to the output word
                char_ = self.vocab.act.i2w[action]
                word.append(char_)
                
        word = u''.join(word)
        return losses, word, action_history


    def beam_search_decode(self, lemma, feats, external_cg=True, unk_avg=True, beam_width=4):
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
        def _valid_actions(encoder):
            valid_actions = []
            if len(encoder) > 1:
                valid_actions += [COPY, DELETE] + self.SUBSTS
            else:
                valid_actions += [END_WORD]
            valid_actions += self.INSERTS
            return valid_actions

        if not external_cg:
            dy.renew_cg()

        # vectorize lemma
        lemma_enc = self._build_lemma(lemma, unk_avg, is_training=False)

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
    
        encoder.pop()  # BEGIN_WORD_CHAR
        
        # a list of tuples:
        #    (decoder state, encoder state, list of previous actions,
        #     log prob of previous actions, log prob of previous actions as dynet expression,
        #     word generated so far)
        beam = [(decoder, encoder, [COPY], 0., 0., [])]

        beam_length = 0
        complete_hypotheses = []
        
        while beam_length <= MAX_ACTION_SEQ_LEN:
            
            if not beam or beam_width == 0:
                break
            
            #if show_oracle_actions:
            #    print 'Action: ', count, self.vocab.act.i2w[action_history[-1]]
            #    print 'Encoder length, char: ', lemma, len(encoder), self.vocab.char.i2w[encoder.s[-1][-1]]
            #    print 'word: ', u''.join(word)
            #    print 'Remaining actions: ', oracle_actions, u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
            #    count += 1
            #elif action_history[-1] >= self.ACT_TRAIN:
            #    print 'Will be adding unseen act embedding: ', self.vocab.act.i2w[action_history[-1]]
            
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            expansion = []
            #print 'Beam length: ', beam_length
            for decoder, encoder, prev_actions, log_p, log_p_expr, word in beam:
                #print 'Expansion: ', action2string(prev_actions, self.vocab), log_p, ''.join(word)
                valid_actions = _valid_actions(encoder)
                # decoder
                decoder_input = dy.concatenate([encoder.embedding(),
                                                features,
                                                self.ACT_LOOKUP[prev_actions[-1]]
                                               ])
                decoder = decoder.add_input(decoder_input)
                # classifier
                if self.double_feats:
                    classifier_input = dy.concatenate([decoder.output(), features])
                else:
                    classifier_input = decoder.output()
                if self.MLP_DIM:
                    h = self.NONLIN(W_s2h * classifier_input + b_s2h)
                else:
                    h = classifier_input
                logits = W_act * h + b_act
                log_probs_expr = dy.log_softmax(logits, valid_actions)
                log_probs = log_probs_expr.npvalue()
                top_actions = np.argsort(log_probs)[-beam_width:]
                #print 'top_actions: ', top_actions, action2string(top_actions, self.vocab) 
                #print 'log_probs: ', log_probs
                #print
                expansion.extend((
                    (decoder, encoder.copy(),
                     list(prev_actions), a, log_p + log_probs[a],
                     log_p_expr + log_probs_expr[a], list(word)) for a in top_actions))

            #print 'Overall, {} expansions'.format(len(expansion))
            beam = []
            expansion.sort(key=lambda e: e[4])
            for e in expansion[-beam_width:]:
                decoder, encoder, prev_actions, action, log_p, log_p_expr, word = e
            
                prev_actions.append(action)

                # execute the action to update the transducer state
                if action == END_WORD:
                    # 1. Finish transduction:
                    #  * beam width should be decremented
                    #  * expansion should be taken off the beam and
                    # stored to final hypotheses set
                    beam_width -= 1
                    complete_hypotheses.append((log_p, log_p_expr, u''.join(word), prev_actions))
                else:
                    if action == COPY:
                        # 1. Increment attention index
                        char_ = encoder.pop()
                        # 2. Append copied character to the output word
                        word.append(self.vocab.char.i2w[char_])
                    elif action == DELETE:               
                        # 1. Increment attention index
                        encoder.pop()
                    elif action in self.SUBSTS:
                        # 1. Append new character to the output word
                        char_ = self.subst_i2w[action]
                        word.append(char_)
                        # 2. Pop replaced character
                        encoder.pop()
                    else:
                        # one of the INSERT actions
                        assert action in self.INSERTS, (
                                u'action: {} ({}), SUBSTS: {}, INSERTS: {}'.format(
                                action, self.vocab.act.i2w[action], self.SUBSTS, self.INSERTS))
                        # 1. Append inserted character to the output word
                        char_ = self.vocab.act.i2w[action]
                        word.append(char_)
                    beam.append((decoder, encoder, prev_actions, log_p, log_p_expr, word))
            
            beam_length += 1

        complete_hypotheses.sort(key=lambda h: h[0], reverse=True)
        #print u'Complete hypotheses:'
        #for log_p, word, actions in complete_hypotheses:
        #    print u'Actions {}, word {}, log p {:.3f}'.format(action2string(actions, self.vocab), word, log_p)
            
        return complete_hypotheses