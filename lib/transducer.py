from __future__ import division

import dynet as dy
import numpy as np

from defaults import (STEP, COPY, DELETE, BEGIN_WORD, END_WORD, UNK,
                      END_WORD_CHAR, MAX_ACTION_SEQ_LEN)
from stack_lstms import Encoder
from datasets import action2string

NONLINS = {'tanh' : dy.tanh, 'ReLU' : dy.rectify}

class Transducer(object):
    def __init__(self, model, vocab, char_dim=100, action_dim=100, feat_dim=20,
                 enc_hidden_dim=200, enc_layers=1, dec_hidden_dim=200, dec_layers=1,
                 vanilla_lstm=False, mlp_dim=0, nonlin='ReLU', lucky_w=55,
                 double_feats=False, param_tying=False, pos_emb=True, 
                 avm_feat_format=False, **kwargs):
        
        self.CHAR_DIM       = char_dim
        self.ACTION_DIM     = action_dim
        self.FEAT_DIM       = feat_dim
        self.ENC_HIDDEN_DIM = enc_hidden_dim
        self.ENC_LAYERS     = enc_layers
        self.DEC_HIDDEN_DIM = dec_hidden_dim
        self.DEC_LAYERS     = dec_layers
        self.LSTM           = dy.VanillaLSTMBuilder if vanilla_lstm else dy.CoupledLSTMBuilder
        self.MLP_DIM        = mlp_dim
        self.NONLIN         = NONLINS.get(nonlin, 'ReLU')
        self.LUCKY_W        = lucky_w
        self.double_feats   = double_feats
        self.param_tying    = param_tying
        self.pos_emb        = pos_emb
        self.avm_feat_format = avm_feat_format

        self.vocab = vocab

        # indices separating train elements from dev/test elements
        self.NUM_CHARS = self.vocab.char_train
        self.NUM_FEATS = self.vocab.feat_train
        self.NUM_POS   = self.vocab.pos_train
        self.NUM_ACTS  = self.vocab.act_train
        # an enumeration of all encoded insertions
        self.INSERTS   = range(self.vocab.number_specials, self.NUM_ACTS)
        
        # report stats
        print u'{} actions: {}'.format(self.NUM_ACTS,
            u', '.join(self.vocab.act.keys()))
        print u'{} features: {}'.format(self.NUM_FEATS,
            u', '.join(self.vocab.feat.keys()))
        print u'{} lemma chars: {}'.format(self.NUM_CHARS,
            u', '.join(self.vocab.char.keys()))

        if self.avm_feat_format:
            self.NUM_FEAT_TYPES = self.vocab.feat_type_train
            print u'{} feature types: {}'.format(self.NUM_FEAT_TYPES,
                u', '.join(self.vocab.feat_type.keys()))
            if self.pos_emb:
                print 'Assuming AVM features, therefore no specialized pos embedding'
                self.pos_emb = False
        
        self._build_model(model)
        # for printing
        self.hyperparams = {'CHAR_DIM'       : self.CHAR_DIM,
                            'FEAT_DIM'       : self.FEAT_DIM,
                            'ACTION_DIM'     : self.ACTION_DIM if not self.param_tying else self.CHAR_DIM,
                            'ENC_HIDDEN_DIM' : self.ENC_HIDDEN_DIM,
                            'ENC_LAYERS'     : self.ENC_LAYERS,
                            'DEC_HIDDEN_DIM' : self.DEC_HIDDEN_DIM,
                            'DEC_LAYERS'     : self.DEC_LAYERS,
                            'LSTM'           : self.LSTM,
                            'MLP_DIM'        : self.MLP_DIM,
                            'NONLIN'         : self.NONLIN,
                            'PARAM_TYING'    : self.param_tying,
                            'POS_EMB'        : self.pos_emb,
                            'AVM_FEATS'      : self.avm_feat_format}

    def _features(self, model):
        # trainable embeddings for characters and actions
        self.CHAR_LOOKUP = model.add_lookup_parameters((self.NUM_CHARS, self.CHAR_DIM))
        if self.param_tying:
            self.ACT_LOOKUP = self.CHAR_LOOKUP
            print 'NB! Using parameter tying: Chars and actions share embedding matrix.'
        else:
            self.ACT_LOOKUP  = model.add_lookup_parameters((self.NUM_ACTS, self.ACTION_DIM))
        # embed features or bag-of-word them?
        if not self.FEAT_DIM:
            print 'Using an n-hot representation for features.'
            # n-hot POS features are simply concatenated to feature vector
            self.FEAT_INPUT_DIM = self.NUM_FEATS + self.NUM_POS
        else:
            self.FEAT_LOOKUP = model.add_lookup_parameters((self.NUM_FEATS, self.FEAT_DIM))
            if self.pos_emb:
                self.POS_LOOKUP = model.add_lookup_parameters((self.NUM_POS, self.FEAT_DIM))
                # POS feature is the only feature with many values (=`self.NUM_POS`), hence + 1.
                # All other features are binary (e.g. SG and PL are disjoint binary features).
                self.FEAT_INPUT_DIM = self.NUM_FEATS*self.FEAT_DIM  # + 1 for POS and - 1 for UNK
                print 'All feature-value pairs are taken to be atomic, except for POS.'
            else:
                self.POS_LOOKUP = self.FEAT_LOOKUP  # self.POS_LOOKUP is probably not needed
                if self.avm_feat_format:
                    self.FEAT_INPUT_DIM = self.NUM_FEAT_TYPES*self.FEAT_DIM
                    print 'All feature-value pairs are taken to be non-atomic.'
                else:
                    self.FEAT_INPUT_DIM = (self.NUM_FEATS - 1)*self.FEAT_DIM  # -1 for UNK
                    print 'Every feature-value pair is taken to be atomic.'

        # BiLSTM encoding lemma
        self.fbuffRNN  = self.LSTM(self.ENC_LAYERS, self.CHAR_DIM, self.ENC_HIDDEN_DIM, model)
        self.bbuffRNN  = self.LSTM(self.ENC_LAYERS, self.CHAR_DIM, self.ENC_HIDDEN_DIM, model)

        # LSTM representing generated word
        self.WORD_REPR_DIM = self.ENC_HIDDEN_DIM*2 + self.ACTION_DIM + self.FEAT_INPUT_DIM
        self.wordRNN  = self.LSTM(self.DEC_LAYERS, self.WORD_REPR_DIM, self.DEC_HIDDEN_DIM, model)
        
        self.CLASSIFIER_IMPUT_DIM = self.DEC_HIDDEN_DIM
        if self.double_feats:
            self.CLASSIFIER_IMPUT_DIM += self.FEAT_INPUT_DIM

        print ' * LEMMA biLSTM:      IN-DIM: {}, OUT-DIM: {}'.format(2*self.CHAR_DIM, 2*self.ENC_HIDDEN_DIM)
        print ' * WORD LSTM:         IN-DIM: {}, OUT-DIM: {}'.format(self.WORD_REPR_DIM, self.DEC_HIDDEN_DIM)
        print ' LEMMA LSTMs have {} layer(s)'.format(self.ENC_LAYERS)
        print ' WORD LSTM has {} layer(s)'.format(self.DEC_LAYERS)
        print
        print ' * CHAR EMBEDDINGS:   IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_CHARS, self.CHAR_DIM)
        if not self.param_tying:
            print ' * ACTION EMBEDDINGS: IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_ACTS, self.ACTION_DIM)
        if self.FEAT_DIM:
            print ' * FEAT. EMBEDDINGS:  IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_FEATS, self.FEAT_DIM)

            
    def _classifier(self, model):
        # single-hidden-layer classifier that works on feature presentation
        # from "self._features"
        if self.MLP_DIM:
            self.pW_s2h = model.add_parameters((self.MLP_DIM, self.CLASSIFIER_IMPUT_DIM))
            self.pb_s2h = model.add_parameters(self.MLP_DIM)
            feature_dim = self.MLP_DIM
            print ' * HIDDEN LAYER:      IN-DIM: {}, OUT-DIM: {}'.format(self.CLASSIFIER_IMPUT_DIM, feature_dim)
        else:
            feature_dim = self.CLASSIFIER_IMPUT_DIM
        # hidden to action
        self.pW_act = model.add_parameters((self.NUM_ACTS, feature_dim))
        self.pb_act = model.add_parameters(self.NUM_ACTS)
        print ' * SOFTMAX:           IN-DIM: {}, OUT-DIM: {}'.format(feature_dim, self.NUM_ACTS)
        
    def _build_model(self, model):
        # feature model
        self._features(model)
        # classifier
        self._classifier(model)
        
    def _build_lemma(self, lemma, unk_avg, is_training):
        # returns a list of character embedding for the lemma
        if is_training:
            lemma_enc = [self.CHAR_LOOKUP[c] for c in lemma]
        else:
            # then vectorize lemma with UNK
            if unk_avg:
                # UNK embedding is the average of trained embeddings (excluding UNK symbol=0)
                UNK_CHAR_EMB = dy.average([self.CHAR_LOOKUP[i] for i in xrange(1, self.NUM_CHARS)])
            else:
                # @TODO Pretrain it with "word dropout", otherwise
                # these are randomly initialized embeddings.
                UNK_CHAR_EMB = self.CHAR_LOOKUP[UNK]
            lemma_enc = [self.CHAR_LOOKUP[c] if c < self.NUM_CHARS else UNK_CHAR_EMB for c in lemma]
        return lemma_enc

    def _build_features(self, pos, feats):
        # represent morpho-syntactic features:
        if self.FEAT_DIM:
            feat_vecs = []
            if self.pos_emb:
                # POS gets a special treatment
                if pos < self.NUM_POS:
                    pos_emb = self.POS_LOOKUP[pos]
                else:
                    pos_emb = self.FEAT_LOOKUP[UNK]
                feat_vecs.append(pos_emb)
            #
            if self.avm_feat_format:
                for ftype in range(self.NUM_FEAT_TYPES):
                    # each feature types gets represented with an embedding
                    # first, check if this feature type in `feats`
                    feat = feats.get(ftype, UNK)
                    # second, check if `feat` seen in training
                    if feat >= self.NUM_FEATS:
                        feat = UNK
                    feat_vecs.append(self.FEAT_LOOKUP[feat])
            #
            else:                
                for feat in range(1, self.NUM_FEATS):  # skip UNK
                    if feat in feats:  # set of indices
                        feat_vecs.append(self.FEAT_LOOKUP[feat])
                    else:
                        feat_vecs.append(self.FEAT_LOOKUP[UNK])
            #
            feats_enc = dy.concatenate(feat_vecs)
        else:
            # (upweighted) bag-of-features
            nhot = np.zeros(self.FEAT_INPUT_DIM)
            nhot[feats] = 1.
            if pos != UNK:
                # simply ignore UNK POS tag
                nhot[pos + self.NUM_FEATS] = 1.
            feats_enc = dy.inputVector(nhot * self.LUCKY_W)

        return feats_enc
    
    def set_dropout(self, dropout):
        self.wordRNN.set_dropout(dropout)

    def disable_dropout(self):
        self.wordRNN.disable_dropout()
    
    def l2_norm(self, with_embeddings=True):
        # specify regularization term: sum of Frobenius/L2-normalized weights
        # assume that we add to a computation graph
        reg = []
        # RNN weight matrices
        for rnn in (self.fbuffRNN, self.bbuffRNN, self.wordRNN):
            for exp in (e for layer in rnn.get_parameter_expressions() for e in layer):
                if len(exp.dim()[0]) != 1:
                    # this is not a bias term
                    reg.append(dy.l2_norm(exp))
        # classifier weight matices
        reg.append(dy.l2_norm(self.pW_act.expr()))
        if self.MLP_DIM:
            reg.append(dy.l2_norm(self.pW_s2h.expr()))
        if with_embeddings:
            # add embedding params
            reg.append(dy.l2_norm(self.FEAT_LOOKUP.expr()))
            reg.append(dy.l2_norm(self.CHAR_LOOKUP.expr()))
            if not self.param_tying:
                reg.append(dy.l2_norm(self.ACT_LOOKUP.expr()))
        return 0.5 * dy.esum(reg)

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

        # add classifier to computation graph
        if self.MLP_DIM:
            # decoder output to hidden
            W_s2h = dy.parameter(self.pW_s2h)
            b_s2h = dy.parameter(self.pb_s2h)
        # hidden to action
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)

        # encoder is a stack which pops lemma characters and their
        # representations from the top. Thus, to get lemma characters
        # in the right order, the lemma has to be reversed.
        encoder.transduce(lemma_enc, lemma)

        encoder.pop()  # BEGIN_WORD_CHAR
        action_history = [COPY]
        word = []
        losses = []
        count = 0

        if show_oracle_actions:
            print
            print u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
            print u''.join([self.vocab.char.i2w[a] for a in lemma])
        
        while len(action_history) <= MAX_ACTION_SEQ_LEN:
            
            if show_oracle_actions:
                print 'Action: ', count, self.vocab.act.i2w[action_history[-1]]
                print 'Encoder length, char: ', lemma, len(encoder), self.vocab.char.i2w[encoder.s[-1][-1]]
                print 'word: ', u''.join(word)
                print 'Remaining actions: ', oracle_actions, u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
                count += 1
            #elif action_history[-1] >= self.NUM_ACTS:
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
                valid_actions += [COPY, DELETE]
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
        #     log prob of previous actions, log prob of previous actions as dynet object,
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
            #elif action_history[-1] >= self.NUM_ACTS:
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
                    else:
                        # one of the INSERT actions
                        assert action in self.INSERTS
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
        #print u'Complete hypotheses:'
        #for log_p, _, word, actions in complete_hypotheses:
        #    print u'Actions {}, word {}, log p {:.3f}'.format(action2string(actions, self.vocab), word, log_p)
            
        return complete_hypotheses