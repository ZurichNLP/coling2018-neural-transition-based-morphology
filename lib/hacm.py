from __future__ import division

import dynet as dy
import numpy as np

import datasets
from defaults import (STEP, BEGIN_WORD, END_WORD, UNK, MAX_ACTION_SEQ_LEN, UNK_CHAR)
from stack_lstms import Encoder
from transducer import Transducer
from datasets import action2string

COPY = -1

class MinimalTransducer(Transducer):
    
    def _classifier(self, model):
        # Here, we add a copy logistic classifier.
        # input + decoder output for good measure
        gen_dim = self.WORD_REPR_DIM + self.CLASSIFIER_IMPUT_DIM
        self.pW_gen = model.add_parameters((1, gen_dim))
        self.pb_gen = model.add_parameters(1)
        print ' * COPY LOGISTIC:     IN-DIM: {}, OUT-DIM: {}'.format(gen_dim, 1)
        super(MinimalTransducer, self)._classifier(model)
    
    def transduce(self, lemma, feats, oracle_actions=None, external_cg=True, sampling=False, unk_avg=True):
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

        show_oracle_actions = False

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
        # decoder output to copy
        W_gen = dy.parameter(self.pW_gen)
        b_gen = dy.parameter(self.pb_gen)
        
#        encoder.pop()  # BEGIN_WORD_CHAR
        action_history = [BEGIN_WORD]
        word = []
        losses = []
        count = 0
        
        if show_oracle_actions:
            print
            if oracle_actions: print u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
            print u''.join([self.vocab.char.i2w[a] for a in lemma])
        
        while len(action_history) <= MAX_ACTION_SEQ_LEN:
            
            encoder_embedding, char_enc = encoder.embedding(extra=True)
            copy_action = self.vocab.act.w2i[self.vocab.char.i2w[char_enc]
                                             if self.vocab.char.i2w[char_enc] in self.vocab.act else UNK_CHAR]
            
            if show_oracle_actions:
                print 'Previous action: ', action_history[-1], self.vocab.act.i2w[action_history[-1]]
                print 'Encoder last ind, char: ', lemma, char_enc, self.vocab.char.i2w[char_enc]
                print 'word: ', u''.join(word)
                print 'Copy action ind, char: ', copy_action, self.vocab.act.i2w[copy_action]
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
            # copy switch
            gen = dy.logistic(W_gen * dy.concatenate([decoder_output, decoder_input]) + b_gen)
            
            logits = W_act * h + b_act
            probs_gen = dy.softmax(logits)
#            valid_actions = np.ones(self.NUM_ACTS) * -np.inf
#            valid_actions[_valid_actions(encoder)] = 0.
#            valid_actions = dy.inputTensor(valid_actions)
#            probs_gen = dy.softmax(logits + valid_actions)
            # log_probs = dy.log_softmax(logits, valid_actions)
            
            if not char_enc < self.NUM_CHARS: #unk char in lemma
                encoder.pop()
                action = STEP
                losses.append(-dy.log(dy.pick(probs_gen, action)))
                action_history.append(action)
                char_ = self.vocab.char.i2w[char_enc]
                word.append(char_)
                
            else:
                probs_copy = np.zeros(self.NUM_ACTS)
#                if copy_action in _valid_actions(encoder):
#                    probs_copy[copy_action] = 1.
                probs_copy[copy_action] = 1.
                probs_copy = dy.inputTensor(probs_copy)
                
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
                if action == STEP:
                    # 1. Increment attention index
                    if char_enc != END_WORD:
                        encoder.pop()
                elif action == END_WORD:
                    # 1. Finish transduction
                    break
                else:
                    # one of the INSERT actions
#                    assert action in self.INSERTS
#                    if action not in self.INSERTS:
#                        print action, self.vocab.act.i2w[action], u''.join([self.vocab.char.i2w[a] for a in lemma]), u''.join(word)
                    # 1. Append inserted character to the output word
                    char_ = self.vocab.act.i2w[action]
                    word.append(char_)
                
        word = u''.join(word)
        return losses, word, action_history

    def beam_search_decode(self, lemma, feats, external_cg=True, unk_avg=True, beam_width=4):
    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    
        if not external_cg:
            dy.renew_cg()
        
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
        # decoder output to copy
        W_gen = dy.parameter(self.pW_gen)
        b_gen = dy.parameter(self.pb_gen)
        
        # a list of tuples:
        #    (decoder state, encoder state, list of previous actions,
        #     log prob of previous actions, log prob of previous actions as dynet object,
        #     word generated so far)
        beam = [(decoder, encoder, [BEGIN_WORD], 0., 0., [])]
        
        beam_length = 0
        complete_hypotheses = []
        
        while beam_length <= MAX_ACTION_SEQ_LEN:
            
            if not beam or beam_width == 0:
                break
        
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            expansion = []
            # print 'Beam length: ', beam_length
            for decoder, encoder, prev_actions, log_p, log_p_expr, word in beam:
                # print 'Expansion: ', action2string(prev_actions, self.vocab), log_p, ''.join(word)
                encoder_embedding, char_enc = encoder.embedding(extra=True)
                copy_action = self.vocab.act.w2i[self.vocab.char.i2w[char_enc] if self.vocab.char.i2w[char_enc] in self.vocab.act.keys() else UNK_CHAR]
                # decoder
                decoder_input = dy.concatenate([encoder_embedding, features, self.ACT_LOOKUP[prev_actions[-1]]])
                decoder = decoder.add_input(decoder_input)
                decoder_output = decoder.output()
                # generate
                if self.MLP_DIM:
                    h = self.NONLIN(W_s2h * decoder_output + b_s2h)
                else:
                    h = decoder_output
                # copy switch
                gen = dy.logistic(W_gen * dy.concatenate([decoder_output, decoder_input]) + b_gen)
                
                logits = W_act * h + b_act
                probs_gen = dy.softmax(logits)
                
                if not char_enc < self.NUM_CHARS:
                    log_probs_expr = dy.log(probs_gen)
                    log_probs = log_probs_expr.npvalue()
                    top_actions = [STEP]
                else:
                    probs_copy = np.zeros(self.NUM_ACTS)
                    probs_copy[copy_action] = 1.
                    probs_copy = dy.inputTensor(probs_copy)
                    probs = dy.cmult(gen, probs_gen) + dy.cmult(1-gen, probs_copy)
                    log_probs_expr = dy.log(probs)
                    log_probs = log_probs_expr.npvalue()
                    top_actions = np.argsort(log_probs)[-beam_width:]
#                print 'top_actions: ', top_actions, action2string(top_actions, self.vocab)
#                print 'log_probs: ', log_probs
#                print
                expansion.extend(( (decoder, encoder.copy(), list(prev_actions), a, log_p + log_probs[a], log_p_expr + log_probs_expr[a], list(word), char_enc) for a in top_actions))
        
            #            print 'Overall, {} expansions'.format(len(expansion))
            beam = []
            expansion.sort(key=lambda e: e[4])
            for e in expansion[-beam_width:]:
                decoder, encoder, prev_actions, action, log_p, log_p_expr, word, char_enc = e
                
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
                    if action == STEP:
                        # 1. Increment attention index and write UNK char
                        if not char_enc < self.NUM_CHARS: # unk char in lemma
                            encoder.pop()
                            char_ = self.vocab.char.i2w[char_enc]
                            word.append(char_)
                        else:
                            # 1. Increment attention index
                            if char_enc != END_WORD:
                                encoder.pop()
                    else:
                        # one of the INSERT actions
                        # 1. Append inserted character to the output word
                        char_ = self.vocab.act.i2w[action]
                        word.append(char_)
                    beam.append((decoder, encoder, prev_actions, log_p, log_p_expr, word))
            beam_length += 1
    
        if not complete_hypotheses:
            # nothing found because the model is so crappy
            complete_hypotheses = [(log_p, log_p_expr, u''.join(word), prev_actions)
                               for _, _, prev_actions, log_p, log_p_expr, word in beam]
            
        complete_hypotheses.sort(key=lambda h: h[0], reverse=True)
#        print u'Complete hypotheses:'
#        for log_p, _, word, actions in complete_hypotheses:
#            print u'Actions {}, word {}, log p {:.3f}'.format(action2string(actions, self.vocab), word, log_p)

        return complete_hypotheses
