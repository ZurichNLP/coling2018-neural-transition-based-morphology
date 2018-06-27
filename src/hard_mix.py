"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2017 shared task data
files and evaluation script.

Usage:
  hard_mix.py [--dynet-seed SEED] [--dynet-mem MEM] [--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION]
  [--patience=PATIENCE] [--reg=REGULARIZATION] [--learning=LEARNING] [--plot]
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
  --patience=PATIENCE           maximum patience [Default: 10]
  --reg=REGULARIZATION          regularization parameter for optimization
  --learning=LEARNING           learning rate parameter for optimization
  --plot                        draw a learning curve plot while training each model
  --beam-width=BEAM_WIDTH       beam search width (default beam width of 1, i.e. greedy search)
  --nbest=NBEST                 run evaluation without training and output nbest results
  --eval                        short-hand for --nbest=1
  --detect_feat_type            detect feature types using knowledge from UniMorph Schema
  --align_smart                 align with Chinese restaurant process like in the paper
  --align_dumb                  align by padding the shortest string (lemma or inflected word)
  --align_leven                 align with levenshtein distance algorithm
  --test_path=TEST_PATH         test set path
"""

import dynet as pc
import numpy as np
from docopt import docopt
from hard import HardAttentionMED, STEP, ALIGN_SYMBOL, EPSILON
from system_base import MAX_PREDICTION_LEN, BEGIN_WORD, END_WORD, UNK
import common

class LeanAttentionMED(HardAttentionMED):


    def build_decoder(self):

        # define softmax
        alphabet_len = len(self.alphabet)
        self.R = self.model.add_parameters((alphabet_len, self.hidden_dim))
        self.bias = self.model.add_parameters(alphabet_len)

        # 2 * HIDDEN_DIM + input_dim, as it gets BLSTM[i], previous output
        concatenated_input_dim = (2*self.hidden_dim + self.input_dim
                                  + len(self.feature_types) * self.feat_input_dim)
        self.decoder_rnn = pc.LSTMBuilder(self.layers,
                                          concatenated_input_dim,
                                          self.hidden_dim,
                                          self.model)

        self.W_p_gen = self.model.add_parameters((1, concatenated_input_dim + self.hidden_dim))
        self.bias_p_gen = self.model.add_parameters(1)

        print 'decoder lstm dimensions are {} x {}'.format(concatenated_input_dim,
                                                           self.hidden_dim)



    def compute_probs(self, decoder_rnn_output, R, bias, W_p_gen, bias_p_gen,
                      input_char, prev_output_vec, blstm_outputs, feats_input, i):

        probs_gen = pc.softmax(R * decoder_rnn_output + bias)

        sigmoid_input = pc.concatenate([prev_output_vec, decoder_rnn_output,
                                        blstm_outputs[i], feats_input])
        p_gen = pc.logistic(W_p_gen * sigmoid_input + bias_p_gen)
        probs_copy_np = np.zeros(len(self.alphabet))
        probs_copy_np[self.alphabet_index[input_char]] = 1
        probs_copy = pc.inputTensor(probs_copy_np)
        probs = pc.cmult(p_gen, probs_gen) + pc.cmult((1-p_gen), probs_copy)
        return probs


    def one_word_loss(self, sample):

        pc.renew_cg()
        R = pc.parameter(self.R)
        bias = pc.parameter(self.bias)
        W_p_gen = pc.parameter(self.W_p_gen)
        bias_p_gen = pc.parameter(self.bias_p_gen)

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
                probs = self.compute_probs(decoder_rnn_output, R, bias, W_p_gen, bias_p_gen,
                                           padded_lemma[i], prev_output_vec,
                                           blstm_outputs, feats_input, i)

                # compute local loss
                loss.append(-pc.log(pc.pick(probs, self.alphabet_index[END_WORD])))
                continue

            # initially, if there is no prefix in the output (shouldn't delay on current input), step forward
            # TODO: check if can remove this condition entirely by adding '<' to both the aligned lemma/word
            if padded_lemma[i] == BEGIN_WORD and aligned_lemma[align_index] != ALIGN_SYMBOL:

                # perform rnn step
                s = s.add_input(decoder_input)
                decoder_rnn_output = s.output()
                probs = self.compute_probs(decoder_rnn_output, R, bias, W_p_gen, bias_p_gen,
                                           padded_lemma[i], prev_output_vec,
                                           blstm_outputs, feats_input, i)

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
                probs = self.compute_probs(decoder_rnn_output, R, bias, W_p_gen, bias_p_gen,
                                           padded_lemma[i], prev_output_vec,
                                           blstm_outputs, feats_input, i)

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
                probs = self.compute_probs(decoder_rnn_output, R, bias, W_p_gen, bias_p_gen,
                                           padded_lemma[i], prev_output_vec,
                                           blstm_outputs, feats_input, i)

                # compute local loss for the step action
                loss.append(-pc.log(pc.pick(probs, self.alphabet_index[STEP])))

                # prepare for the next iteration - "feedback"
                prev_output_vec = self.char_lookup[self.alphabet_index[STEP]]
                i += 1

        # loss = esum(loss)
        loss = pc.average(loss)
        return loss


    def predict_output_sequence(self, lemma, feats,
                                termination_condition=MAX_PREDICTION_LEN):

        pc.renew_cg()
        R = pc.parameter(self.R)
        bias = pc.parameter(self.bias)
        W_p_gen = pc.parameter(self.W_p_gen)
        bias_p_gen = pc.parameter(self.bias_p_gen)

        # convert characters to matching embeddings, if UNK handle properly
        # convert features to matching embeddings, if UNK handle properly
        # run encoder etc.
        partially_computed_results = self.run_rnn_partially(lemma, feats)

        # initialize the decoder rnn
        s_0 = self.decoder_rnn.initial_state()
        s = s_0

        num_outputs = 0

        hypos = [self.initialize_hypothesis(s_0, *partially_computed_results)]

        # run the decoder through the sequence and predict characters
        while num_outputs < termination_condition:

            # at each stage:
            # create all expansions from the previous beam:
            new_hypos = []
            for hypothesis in hypos:

                # cant expand finished sequences
                if hypothesis.seq[-1] == END_WORD:
                    new_hypos.append(hypothesis)
                    continue

                # prepare input vector and perform LSTM step
                padded_lemma, feats_input, blstm_outputs = partially_computed_results
                prev_output = hypothesis.seq[-1]
                i = hypothesis.i
                if prev_output in self.alphabet_index:
                    prev_output_vec = self.char_lookup[self.alphabet_index[prev_output]]
                else:
                    prev_output_vec = self.char_lookup[self.alphabet_index[STEP]]

                decoder_input = pc.concatenate([prev_output_vec,
                                                blstm_outputs[i],
                                                feats_input])

                s = hypothesis.decoder_state.add_input(decoder_input)
                decoder_rnn_output = s.output()

                input_char = padded_lemma[i]
                if input_char in self.alphabet:
                    probs = self.compute_probs(decoder_rnn_output, R, bias, W_p_gen, bias_p_gen,
                                               input_char, prev_output_vec,
                                               blstm_outputs, feats_input, i)

                    probs = probs.vec_value()
                    next_char_indeces = (common.argmax(probs, n=self.beam_width)
                                         if self.beam_width > 1
                                         else [common.argmax(probs)])

                    for k in next_char_indeces:
                        new_hypothesis = self.extend_hypothesis(hypothesis,
                                                                probs[k],
                                                                self.alphabet[k],
                                                                s,
                                                                *partially_computed_results)
                        new_hypos.append(new_hypothesis)
                else:
                    # character unseen in training
                    # extend hypothesis with it and this choice is deterministic
                    new_hypothesis = self.extend_hypothesis(hypothesis,
                                                            1.,
                                                            input_char,
                                                            s,
                                                            *partially_computed_results)
                    new_hypos.append(new_hypothesis)


            # add the expansions with the largest probability to the beam
            new_probs = [hypo.prob for hypo in new_hypos]
            argmax_indices = (common.argmax(new_probs, n=self.beam_width) if self.beam_width > 1
                              else [common.argmax(new_probs)])
            hypos = [new_hypos[l] for l in argmax_indices]

            # if we need just one best hypothesis and it is a complete sequence,
            # we're finished with the search.
            if self.nbest == 1 and hypos[0].seq[-1] == END_WORD:
                break

            # if we need n-best hypotheses and all are complete sequences,
            # we can't exapand them anymore and so we're done with the search.
            elif all(hypo.seq[-1] == END_WORD for hypo in new_hypos):
                break

            # go for another round of expansions
            else:
                num_outputs += 1

        if self.nbest < self.beam_width:
            final_probs = [hypo.prob for hypo in new_hypos]
            final_indices = (common.argmax(final_probs, n=self.nbest) if self.nbest > 1
                             else [common.argmax(final_probs)])
        else:
            final_indices = range(self.beam_width)

        # remove the beginning and end word symbol
        return [u''.join(new_hypos[l].seq[1:-1]) for l in final_indices]


if __name__ == "__main__":

    arguments = docopt(__doc__)
    model = LeanAttentionMED(arguments)
    model.fit()
