from __future__ import division

import dynet as dy
import numpy as np

import datasets
from defaults import (COPY, DELETE, BEGIN_WORD, END_WORD, UNK, MAX_ACTION_SEQ_LEN)
from stack_lstms import Encoder, StackRNN, StackBiRNN
from transducer import Transducer

class EditTransducer(Transducer):
    
    def __init__(self, model, vocab, **kwargs):
        
        #self.INSERT = COPY + 1
        # Actually, action types are: COPY, DELETE, INSERT
        #self.NUM_ACT_TYPES = self.INSERT + 1
        #kwargs['action_dim'] = 10
        super(EditTransducer, self).__init__(model, vocab, **kwargs)
        
    #def _classifier(self, model):
        # Here, we add a copy logistic classifier.
        # input + decoder output for good measure
        #op_dim = self.CHAR_DIM + self.ACTION_DIM + self.FEAT_INPUT_DIM
        #self.pW_op = model.add_parameters((1, gen_dim))
        #self.pb_op = model.add_parameters(1)
        #print ' * COPY LOGISTIC:     IN-DIM: {}, OUT-DIM: {}'.format(gen_dim, 1)
        #super(MinimalTransducer, self)._classifier(model)
    
    def _features(self, model):
        # trainable embeddings for characters and actions
        self.CHAR_LOOKUP = model.add_lookup_parameters((self.NUM_CHARS, self.CHAR_DIM))
        self.ACT_LOOKUP  = model.add_lookup_parameters((self.NUM_ACTS, self.ACTION_DIM))
        self.p_empty_string = model.add_parameters(self.CHAR_DIM)
        # embed features or bag-of-word them?
        if not self.FEAT_DIM:
            print 'Using an n-hot representation for features.'
            # n-hot POS features are simply concatenated to feature vector
            self.FEAT_INPUT_DIM = self.NUM_FEATS + self.NUM_POS
        else:
            self.FEAT_LOOKUP = model.add_lookup_parameters((self.NUM_FEATS, self.FEAT_DIM))
            self.POS_LOOKUP = model.add_lookup_parameters((self.NUM_POS, self.FEAT_DIM))
            # POS feature is the only feature with many values (=`self.NUM_POS`), hence + 1.
            # All other features are binary (e.g. SG and PL are disjoint binary features).
            # This is the implementation of Aharoni & Goldberg
            self.FEAT_INPUT_DIM = (self.NUM_FEATS + 1)*self.FEAT_DIM

        # BiLSTM encoding lemma
        self.fbuffRNN  = self.LSTM(self.ENC_LAYERS, self.CHAR_DIM, self.ENC_HIDDEN_DIM, model)
        self.bbuffRNN  = self.LSTM(self.ENC_LAYERS, self.CHAR_DIM, self.ENC_HIDDEN_DIM, model)
        
        # LSTM representing generated word
        self.WORD_REPR_DIM = self.CHAR_DIM + self.ACTION_DIM + self.FEAT_INPUT_DIM
        self.wordRNN  = self.LSTM(self.DEC_LAYERS, self.WORD_REPR_DIM, self.DEC_HIDDEN_DIM, model)
        
        self.CLASSIFIER_IMPUT_DIM = self.DEC_HIDDEN_DIM + self.ENC_HIDDEN_DIM*2 + self.FEAT_INPUT_DIM

        print ' * LEMMA biLSTM:      IN-DIM: {}, OUT-DIM: {}'.format(2*self.CHAR_DIM, 2*self.ENC_HIDDEN_DIM)
        print ' * WORD LSTM:         IN-DIM: {}, OUT-DIM: {}'.format(self.WORD_REPR_DIM, self.DEC_HIDDEN_DIM)
        print ' LEMMA LSTMs have {} layer(s)'.format(self.ENC_LAYERS)
        print ' WORD LSTM has {} layer(s)'.format(self.DEC_LAYERS)
        print
        print ' * CHAR EMBEDDINGS:   IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_CHARS, self.CHAR_DIM)
        print ' * ACTION EMBEDDINGS: IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_ACTS, self.ACTION_DIM)
        if self.FEAT_DIM:
            print ' * FEAT. EMBEDDINGS:  IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_FEATS, self.FEAT_DIM)

    
    def transduce(self, lemma, feats, oracle_actions=None, external_cg=True, sampling=False, unk_avg=True):
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
        def _valid_actions(encoder):
            valid_actions = []
            if len(encoder) > 1:
                valid_actions += [COPY, DELETE]
            else:
                valid_actions += [END_WORD]
            valid_actions += self.INSERTS
            return valid_actions
        
        show_oracle_actions = True

        if not external_cg:
            dy.renew_cg()
        
        if unk_avg:
            # Dealing with unseen data, what UNK embedding for characters
            # and actions do we use? => Average trained embeddings.
            UNK_CHAR_EMB = dy.average([self.CHAR_LOOKUP[i] for i in xrange(1, self.CHAR_TRAIN)])
            #UNK_ACT_EMB  = dy.average([self.ACT_LOOKUP[i] for i in xrange(1, self.ACT_TRAIN)])
        else:
            # @TODO Pretrain it with "word dropout", otherwise
            # these are randomly initialized embeddings.
            UNK_CHAR_EMB = self.CHAR_LOOKUP[UNK]
            #UNK_ACT_EMB  = self.ACT_LOOKUP[UNK]
            
        if oracle_actions:
            # reverse to enable simple popping
            oracle_actions = oracle_actions[::-1]
            oracle_actions.pop()  # COPY of BEGIN_WORD_CHAR          
            # vectorize lemma without UNK
            lemma_enc = [self.CHAR_LOOKUP[c] for c in lemma]            
        else:
            # vectorize lemma with UNK
            lemma_enc = [self.CHAR_LOOKUP[c] if c < self.CHAR_TRAIN else UNK_CHAR_EMB for c in lemma]

        # vectorize features
        features = self._build_features(*feats)
            
        # add encoder and decoder to computation graph
        encoder = Encoder(self.fbuffRNN, self.bbuffRNN)
        decoder = StackRNN(self.wordRNN)
        
        # encoder is a stack which pops lemma characters and their
        # representations from the top. Thus, to get lemma characters
        # in the right order, we reverse the lemma.
        encoder.transduce(lemma_enc, zip(lemma, lemma_enc))

        # add classifier to computation graph
        if self.MLP_DIM:
            # decoder output to hidden
            W_s2h = dy.parameter(self.pW_s2h)
            b_s2h = dy.parameter(self.pb_s2h)
        # hidden to action
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)
        
        # ACTION TYPE EMBEDDING
        #COPY_EMB   = self.ACT_LOOKUP[COPY]
        #DELETE_EMB = self.ACT_LOOKUP[DELETE]
        #INSERT_EMB = self.ACT_LOOKUP[self.INSERT]
        
        #print 'Action embs: ', COPY_EMB.npvalue()
    
        # DETERMINISTICALLY COPY BEGIN_WORD_CHAR
        
        #print 'sizes: ',  len(encoder.s[-1]), len(encoder.s[-1][-1])
        
        empty_string = dy.parameter(self.p_empty_string)

        _, char_emb = encoder.pop()
        action_history = [COPY]
        decoder.push(dy.concatenate([char_emb, self.ACT_LOOKUP[COPY], features]))
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
                #print 'Remaining actions: ', oracle_actions, u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
                count += 1
            #elif action_history[-1] >= self.ACT_TRAIN:
            #    print 'Will be adding unseen act embedding: ', self.vocab.act.i2w[action_history[-1]]
            
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            valid_actions = _valid_actions(encoder)

            # classifier
            classifier_input = dy.concatenate([encoder.embedding(), decoder.embedding(), features])
            
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
                    try:
                        action = np.argmax(log_probs.npvalue())
                    except Exception, e:
                        print 'Encoder length, char: ', ''.join([self.vocab.char.i2w[c] for c in lemma]), len(encoder), self.vocab.char.i2w[encoder.s[-1][-1]]
                        print 'word: ', u''.join(word)
                        print 'valid action: ',  len(valid_actions), valid_actions
                        print 'Actions: ', ''.join([self.vocab.act.i2w[a] for a in action_history])
                        raise e 
            else:
                action = oracle_actions.pop()

            losses.append(dy.pick(log_probs, action))
            action_history.append(action)

            # execute the action to update the transducer state
            if action == COPY:
                # 1. Increment attention index
                char_, char_emb = encoder.pop()
                # 3. Append copied character to the output word
                word.append(self.vocab.char.i2w[char_])
            elif action == DELETE:               
                # 1. Increment attention index
                char_, char_emb = encoder.pop()
                # 2. Add popped character and action to decoder
            elif action == END_WORD:
                # 1. Finish transduction
                break
            else:
                # one of the INSERT actions
                assert action in self.INSERTS
                # 1. Append inserted character to the output word
                char_str = self.vocab.act.i2w[action]
                word.append(char_str)
                char_emb = empty_string  # empty string embedding
            # Add character popped from encoder and action to decoder
            decoder.push(dy.concatenate([char_emb, self.ACT_LOOKUP[action], features]))
                
        word = u''.join(word)
        return losses, word, action_history