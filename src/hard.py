"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2017 shared task data
files and evaluation script.

Usage:
  hard.py [--dynet-seed SEED] [--dynet-mem MEM] [--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION]
  [--reg=REGULARIZATION] [--learning=LEARNING] [--plot]
  [--beam-width=BEAM_WIDTH] [--nbest=NBEST | --eval]
  [--detect_feat_type] [--align_smart | --align_dumb | --align_leven]
  TRAIN_PATH DEV_PATH RESULTS_PATH [--test_path=TEST_PATH]

Arguments:
  TRAIN_PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV_PATH      development set path, possibly relative to "data/all/"
  RESULTS_PATH  results file to be written, possibly relative to "results"

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET
  --input=INPUT                 input vector dimensions
  --hidden=HIDDEN               hidden layer dimensions
  --feat-input=FEAT             feature input vector dimension
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
  --reg=REGULARIZATION          regularization parameter for optimization
  --learning=LEARNING           learning rate parameter for optimization
  --plot                        draw a learning curve plot while training each model
  --beam-width=BEAM_WIDTH       beam search width (default beam width of 1, i.e. greedy search)
  --nbest=NBEST                 run evaluation without training and output nbest results
  --eval                        short-hand for --nbest=1
  --detect_feat_type            detect feature types using knowledge from UniMorph Schema
  --align_smart                 align with Chinese restaurant process like in the paper
  --align_dumb                  align by padding the shortest string (lemma or inflected word)
  --align_leven                 align with `levenshtein` function from of the baseline system
  --test_path=TEST_PATH         test set path
"""


import dynet as pc
import numpy as np
from docopt import docopt
from collections import namedtuple
from system_base import BaseModel, BaseDataSet, MAX_PREDICTION_LEN, BEGIN_WORD, END_WORD
import prepare_sigmorphon_data
import common

EPSILON = '*'
STEP = '^'
ALIGN_SYMBOL = '~'


Hypothesis = namedtuple('Hypothesis', 'seq prob decoder_state i')

class HardDataSet(BaseDataSet):
    # class to hold a dataset
    def __init__(self, words, lemmas, feat_dicts, aligned_pairs=None):
        super(HardDataSet, self).__init__(words, lemmas, feat_dicts)
        self.aligned_pairs = aligned_pairs

    def set_aligned(self, aligned_pairs):
        self.aligned_pairs = aligned_pairs
        self.dataset = self.words, self.lemmas, self.feat_dicts, self.aligned_pairs


class HardAttentionMED(BaseModel):
    
    def __init__(self, arguments):

        super(HardAttentionMED, self).__init__(arguments, HardDataSet)

        if arguments['--align_dumb']:
            self.aligner = common.dumb_align
            aligner_param = 'DUMB'
        elif arguments['--align_leven']:
            import sys
            sys.path.append('../data/baseline')
            from baseline import levenshtein
            
            def leven_aligner(zipped_pairs, align_symbol):
                aligned = []
                for s1, s2 in zipped_pairs:
                    als1, als2, _ = levenshtein(s1, s2)
                    als1 = als1.replace('_', align_symbol)
                    als2 = als2.replace('_', align_symbol)
                    aligned.append((als1, als2))
                return aligned
            
            self.aligner = leven_aligner
            aligner_param = 'LEVENSHTEIN'
        else:
            self.aligner = common.mcmc_align
            aligner_param = 'SMART'

        self.hyper_params['ALIGNER'] = aligner_param
        for param in self.hyper_params:
            print param + '=' + str(self.hyper_params[param])


    def build_alphabet_feature_types(self):
    
        super(HardAttentionMED, self).build_alphabet_feature_types()
        
        # used during decoding
        self.alphabet.append(EPSILON)
        # add indices to alphabet - used to indicate when copying from lemma to word
        #for marker in (str(i) for i in xrange(3 * MAX_PREDICTION_LEN)):
        #    self.alphabet.append(marker)
        # indicates the FST to step forward in the input
        self.alphabet.append(STEP)


    def read_in_data(self):
        
        super(HardAttentionMED, self).read_in_data()
        
        if not self.eval_only:
            # align the words to the inflections, the alignment will later be used by the model
            print 'started aligning'
            train_word_pairs = zip(self.train_data.lemmas, self.train_data.words)
            dev_word_pairs = zip(self.dev_data.lemmas, self.dev_data.words)
            print 'aligning with %s' % self.aligner
            
            train_aligned_pairs = self.aligner(train_word_pairs, ALIGN_SYMBOL)
            self.train_data.set_aligned(train_aligned_pairs)
            
            dev_aligned_pairs = self.aligner(dev_word_pairs, ALIGN_SYMBOL)
            self.dev_data.set_aligned(dev_aligned_pairs)
            print 'finished aligning'
            
            
    def build_decoder(self):
        
        # define softmax
        super(HardAttentionMED, self).build_decoder()

        # 2 * HIDDEN_DIM + input_dim, as it gets BLSTM[i], previous output
        concatenated_input_dim = (2 * self.hidden_dim + self.input_dim
                                  + len(self.feature_types) * self.feat_input_dim)
        self.decoder_rnn = pc.LSTMBuilder(self.layers,
                                          concatenated_input_dim,
                                          self.hidden_dim,
                                          self.model)
        print 'decoder lstm dimensions are {} x {}'.format(concatenated_input_dim,
                                                           self.hidden_dim)


    def run_rnn_partially(self, lemma, feats):
        
        padded_lemma = BEGIN_WORD + lemma + END_WORD
        # convert characters to matching embeddings
        lemma_char_vecs = self.encode_lemma(padded_lemma)
        # convert features to matching embeddings, if UNK handle properly
        feat_vecs = self.encode_feats(feats)
        feats_input = pc.concatenate(feat_vecs)
        blstm_outputs = self.bilstm_transduce(lemma_char_vecs)
        return padded_lemma, feats_input, blstm_outputs
    
    
    def make_hypothesis(self, seq, prob, decoder_state, i):
        return Hypothesis(seq, prob, decoder_state, i)


    def initialize_hypothesis(self, s_0, *args, **kwargs):
        return self.make_hypothesis(seq=[BEGIN_WORD], prob=0.0, decoder_state=s_0, i=0)


    def extend_hypothesis(self, hypothesis, new_prob, new_char, s, *args, **kwargs):
        new_prob = hypothesis.prob + np.log(new_prob)
        new_seq = hypothesis.seq + [new_char]
        i = hypothesis.i
        padded_lemma = args[0]        
        if new_char == STEP or new_char not in self.alphabet:
            if i < len(padded_lemma) - 1:
                i += 1
        return self.make_hypothesis(seq=new_seq, prob=new_prob, decoder_state=s, i=i)
    

    def compute_decoder_input(self, hypothesis, *args):
        _, feats_input, blstm_outputs = args
        # prepare input vector and perform LSTM step
        prev_output_vec = self.char_lookup[self.alphabet_index[hypothesis.seq[-1]]]
        decoder_input = pc.concatenate([prev_output_vec,
                                        blstm_outputs[hypothesis.i],
                                        feats_input])
        return decoder_input


    def produce_word_from_prediction(self, predicted_sequence):
        return predicted_sequence.replace(STEP, '')


    def one_word_loss(self, sample):
        
        pc.renew_cg()
        R = pc.parameter(self.R)
        bias = pc.parameter(self.bias)       

        word, lemma, feats, aligned_pair = sample
        padded_lemma, feats_input, blstm_outputs = self.run_rnn_partially(lemma, feats)

        # initialize the decoder rnn
        s_0 = self.decoder_rnn.initial_state()
        s = s_0

        # set prev_output_vec for first lstm step as BEGIN_WORD
        prev_output_vec = self.char_lookup[self.alphabet_index[BEGIN_WORD]]
        loss = []
        
        # i is input index, j is output index
        i = 0
        j = 0

        # go through alignments, progress j when new output is introduced, progress i when new char is seen on lemma (no ~)
        aligned_lemma, aligned_word = aligned_pair
        aligned_lemma += END_WORD
        aligned_word += END_WORD

        # run through the alignments
        for align_index, (input_char, output_char) in enumerate(zip(aligned_lemma, aligned_word)):
            possible_outputs = []

            # feedback, blstm[i], feats
            decoder_input = pc.concatenate([prev_output_vec, blstm_outputs[i], feats_input])

            # if reached the end word symbol
            if output_char == END_WORD:
                s = s.add_input(decoder_input)
                decoder_rnn_output = s.output()
                probs = pc.softmax(R * decoder_rnn_output + bias)

                # compute local loss
                loss.append(-pc.log(pc.pick(probs, self.alphabet_index[END_WORD])))
                continue

            # initially, if there is no prefix in the output (shouldn't delay on current input), step forward
            # TODO: check if can remove this condition entirely by adding '<' to both the aligned lemma/word
            if padded_lemma[i] == BEGIN_WORD and aligned_lemma[align_index] != ALIGN_SYMBOL:

                # perform rnn step
                s = s.add_input(decoder_input)
                decoder_rnn_output = s.output()
                probs = pc.softmax(R * decoder_rnn_output + bias)

                # compute local loss
                loss.append(-pc.log(pc.pick(probs, self.alphabet_index[STEP])))

                # prepare for the next iteration - "feedback"
                prev_output_vec = self.char_lookup[self.alphabet_index[STEP]]
                prev_char_vec = self.char_lookup[self.alphabet_index[EPSILON]]
                i += 1

            # if 0-to-1 or 1-to-1 alignment, compute loss for predicting the output symbol
            if aligned_word[align_index] != ALIGN_SYMBOL:
                decoder_input = pc.concatenate([prev_output_vec, blstm_outputs[i], feats_input])

                # feed new input to decoder
                s = s.add_input(decoder_input)
                decoder_rnn_output = s.output()
                probs = pc.softmax(R * decoder_rnn_output + bias)

                if aligned_word[align_index] in self.alphabet_index:
                    current_loss = -pc.log(pc.pick(probs, self.alphabet_index[aligned_word[align_index]]))

                    # prepare for the next iteration - "feedback"
                    prev_output_vec = self.char_lookup[self.alphabet_index[aligned_word[align_index]]]
                else:
                    current_loss = -pc.log(pc.pick(probs, self.alphabet_index[UNK]))

                    # prepare for the next iteration - "feedback"
                    prev_output_vec = self.char_lookup[self.alphabet_index[UNK]]
                loss.append(current_loss)

                j += 1

            # if the input's not done and the next is not a 0-to-1 alignment, perform step
            if i < len(padded_lemma) - 1 and aligned_lemma[align_index + 1] != ALIGN_SYMBOL:
                # perform rnn step
                # feedback, blstm[i], feats
                decoder_input = pc.concatenate([prev_output_vec, blstm_outputs[i], feats_input])

                s = s.add_input(decoder_input)
                decoder_rnn_output = s.output()
                probs = pc.softmax(R * decoder_rnn_output + bias)

                # compute local loss for the step action
                loss.append(-pc.log(pc.pick(probs, self.alphabet_index[STEP])))

                # prepare for the next iteration - "feedback"
                prev_output_vec = self.char_lookup[self.alphabet_index[STEP]]
                i += 1

        # loss = esum(loss)
        loss = pc.average(loss)
        return loss


if __name__ == "__main__":
    
    arguments = docopt(__doc__)
    model = HardAttentionMED(arguments)
    model.fit()