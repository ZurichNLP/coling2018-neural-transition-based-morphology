"""Base class for MED for inflection generation, using the sigmorphon 2017 shared task data files and evaluation script.

Usage:
  soft_attention.py [--dynet-seed SEED] [--dynet-mem MEM] [--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION]
  [--reg=REGULARIZATION] [--learning=LEARNING] [--plot]
  [--beam-width=BEAM_WIDTH] [--nbest=NBEST | --eval]
  [--detect_feat_type]
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
  --test_path=TEST_PATH         test set path
"""

import traceback
import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import os
import common
import dynet as pc


np.random.seed(17)
random.seed(17)

from docopt import docopt
from collections import defaultdict, namedtuple

import feature_type_detection

# load default values for paths, NN dimensions, some training hyperparams
from defaults import (SRC_PATH, RESULTS_PATH, DATA_PATH,
                      INPUT_DIM, FEAT_INPUT_DIM, HIDDEN_DIM, LAYERS,
                      EPOCHS, OPTIMIZATION, DYNET_MEM)

# additional default values
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADADELTA'
EARLY_STOPPING = True
MAX_PATIENCE = 10 # SC now handled by commandline arguments
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1
PARALLELIZE = True
BEAM_WIDTH = 12

UNK = '#'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'

Hypothesis = namedtuple('Hypothesis', 'seq prob decoder_state')

class BaseDataSet(object):
    # class to hold a dataset
    def __init__(self, words, lemmas, feat_dicts):
        self.words = words
        self.lemmas = lemmas
        self.feat_dicts = feat_dicts
        self.dataset = self.words, self.lemmas, self.feat_dicts
        self.length = len(self.words)

    def iter(self, indices=None, shuffle=False):
        zipped = zip(*self.dataset)
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

    @classmethod
    def from_file(cls, path, detect_feat_type=False, *args, **kwargs):
        # load train, dev, test data
        # This accounts for the possibility to select alternative feature dictionary construction methods
        # this returns a `DataSet` with fields: words, lemmas, feat_dicts
        if detect_feat_type:
            make_feat_dict = feature_type_detection.make_feat_dict
        else:
            make_feat_dict = lambda x, lang: prepare_sigmorphon_data.make_feat_dict(x)
        w, l, f = prepare_sigmorphon_data.load_data(path, make_feat_dict)
        #return #cls(prepare_sigmorphon_data.load_data(path, make_feat_dict))
        return cls(w, l, f, *args, **kwargs)


class BaseModel(object):

    def __init__(self, arguments,
                 dataset=BaseDataSet):

        self.DataSet = dataset
        print arguments

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

        # obligatory params: Paths
        self.train_path = common.check_path(arguments['TRAIN_PATH'], 'TRAIN_PATH')
        self.dev_path = common.check_path(arguments['DEV_PATH'], 'DEV_PATH')
        self.results_file_path = common.check_path(arguments['RESULTS_PATH'],
                                                   'RESULTS_PATH', is_data_path=False)
        # optional params: defaults
        if arguments['--test_path']:
            self.test_path = common.check_path(arguments['--test_path'], 'test_path')
        else:
            # indicates no test set eval should be performed
            self.test_path = None
        if arguments['--input']:
            self.input_dim = int(arguments['--input'])
        else:
            self.input_dim = INPUT_DIM
        if arguments['--hidden']:
            self.hidden_dim = int(arguments['--hidden'])
        else:
            self.hidden_dim = HIDDEN_DIM
        if arguments['--feat-input']:
            self.feat_input_dim = int(arguments['--feat-input'])
        else:
            self.feat_input_dim = FEAT_INPUT_DIM
        if arguments['--epochs']:
            self.epochs = int(arguments['--epochs'])
        else:
            self.epochs = EPOCHS
        if arguments['--layers']:
            self.layers = int(arguments['--layers'])
        else:
            self.layers = LAYERS
        if arguments['--optimization']:
            self.optimization = arguments['--optimization']
        else:
            self.optimization = OPTIMIZATION
        if arguments['--patience']:
            self.patience = arguments['--patience']
        else:
            self.patience = MAX_PATIENCE
        if arguments['--reg']:
            self.regularization = float(arguments['--reg'])
        else:
            self.regularization = REGULARIZATION
        if arguments['--learning']:
            self.learning_rate = float(arguments['--learning'])
        else:
            self.learning_rate = LEARNING_RATE
        if arguments['--plot']:
            self.plot = True
            from matplotlib import pyplot as plt
        else:
            self.plot = False
        # TODO ensemble not implemented
        #if arguments['--ensemble']:
        #    self.ensemble = arguments['--ensemble']
        #else:
        #    self.ensemble = False
            # TODO nbest implies beam search, so
            # (eval_only)              => (no training), greedy search eval, one output
            # (eval_only +) beam_width => (no training), beam search eval, one output
            # Also, nbest implies eval_only, so
            # eval_only + nbest      => beam search eval, nbest outputs with default beam_width
            # eval_only + nbest + beam_width => beam search eval, nbest outputs with beam_width
            #
            # So, flag beam is not needed, beam_width is needed,
            # beam_width = 1 implies greedy search
            # nbest and eval_only are disjoint, eval_only is just a shorthand for nbest=1
        if arguments['--nbest']:
            # this implies eval_only
            self.nbest = int(arguments['--nbest'])
            self.eval_only = True
        elif arguments['--eval']:
            # short-hand for --nbest=1
            self.nbest = 1
            self.eval_only = True
        else:
            # training
            self.nbest = 1
            self.eval_only = False
        if arguments['--beam-width']:
            self.beam_width = int(arguments['--beam-width'])
            if self.beam_width < self.nbest:
                print '`nbest` greater than `beam width`. Setting `beam width` equal to `nbest`.'
                self.beam_width = self.nbest
        else:
            self.beam_width = self.nbest
        if arguments['--detect_feat_type']:
            self.detect_feat_type = True
        else:
            self.detect_feat_type = False

        # TODO: MAX_PREDICTION_LEN, MAX_PATIENCE should be arguments
        self.hyper_params = {'INPUT_DIM': self.input_dim, 'HIDDEN_DIM': self.hidden_dim,
                             'FEAT_INPUT_DIM': self.feat_input_dim, 'EPOCHS': self.epochs,
                             'LAYERS': self.layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                             'OPTIMIZATION': self.optimization, 'PATIENCE': self.patience,
                             'REGULARIZATION': self.regularization, 'LEARNING_RATE': self.learning_rate,
                             'BEAM_WIDTH': self.beam_width, 'FEATURE_TYPE_DETECT': self.detect_feat_type}

        print 'train path = ' + self.train_path
        print 'dev path = ' + self.dev_path
        print 'result path = ' + self.results_file_path
        if self.test_path:
            print 'test path = ' + self.test_path
        else:
            print 'No test set.'
        # TODO Remove the two lines below: This should be done in Children Classes
        #for param in hyper_params:
        #    print param + '=' + str(hyper_params[param])

        # some filenames defined from `self.results_file_path`
        self.log_file_name = self.results_file_path + '_log.txt'
        self.result_file_png = self.results_file_path + '.png'
        self.tmp_model_path = self.results_file_path + '_bestmodel.txt'
        self.result_epochs = self.results_file_path + '.epochs'


    def build_alphabet_feature_types(self):
        self.alphabet, self.feature_types = prepare_sigmorphon_data.get_alphabet(
            self.train_data.words, self.train_data.lemmas, self.train_data.feat_dicts)
        self.alphabet.append(UNK)
        self.alphabet.append(BEGIN_WORD)
        self.alphabet.append(END_WORD)


    def build_alphabet_index(self):
        # char 2 int
        self.alphabet_index = dict(zip(self.alphabet, range(0, len(self.alphabet))))
        # TODO inverse_alphabet_index is not needed since alphabet is a list
        # TODO merge with method above
        #self.inverse_alphabet_index = lambda index: self.alphabet[index]


    def build_feature_alphabet_feat_index(self):
        # feat 2 int
        # TODO why get_feature_alphabet in common and get_alphabet in prepare_sigmorphon_data?
        self.feature_alphabet = common.get_feature_alphabet(self.train_data.feat_dicts)
        self.feature_alphabet.append(UNK_FEAT)
        self.feat_index = dict(zip(self.feature_alphabet, range(0, len(self.feature_alphabet))))


    def read_in_data(self):
        # load data
        #(words, lemmas, feat_dicts)
        self.train_data = self.DataSet.from_file(self.train_path, self.detect_feat_type)
        self.dev_data = self.DataSet.from_file(self.dev_path, self.detect_feat_type)
        if self.test_path:
            self.test_data = self.DataSet.from_file(self.test_path, self.detect_feat_type)

        # build character alphabet, including all characters in train data as well as special symbols
        # like UNK for unknown character, BEGIN_WORD, END_WORD, etc.
        self.build_alphabet_feature_types()
        # build a 1-to-1 map from chars of alphabet to integers and its inverse
        self.build_alphabet_index()
        # build feature alphabet (=all feature-value pairs seen in train data) and an indexing map
        self.build_feature_alphabet_feat_index()


    def build_model(self):
        print 'creating model...'
        self.model = pc.Model()

        alphabet_len = len(self.alphabet)
        # character embeddings:
        # This layer is shared between encoder and decoder
        self.char_lookup = self.model.add_lookup_parameters((alphabet_len,
                                                             self.input_dim))

        # feature embeddings:
        # This layer is used in both soft & hard attention models to embed features.
        # So, feature embeddings live in a different space but of the same size.
        self.feat_lookup = self.model.add_lookup_parameters((len(self.feature_alphabet),
                                                             self.feat_input_dim))

        # needs to be defined in Children classes: encoder / decoder
        self.build_encoder()
        self.build_decoder()
        print 'finished creating model'


    def encode_feats(self, feat_dict):
        feat_vecs = []
        for feat in sorted(self.feature_types):
            # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
            # if this feature has a value, take it from the lookup. otherwise use UNK
            if feat in feat_dict:
                feat_str = feat + ':' + feat_dict[feat]
                try:
                    feat_vecs.append(self.feat_lookup[self.feat_index[feat_str]])
                except KeyError:
                    # handle UNK or dropout
                    feat_vecs.append(self.feat_lookup[self.feat_index[UNK_FEAT]])
            else:
                feat_vecs.append(self.feat_lookup[self.feat_index[UNK_FEAT]])
        return feat_vecs


    def encode_lemma(self, lemma):
        # lemma is NOT PADDED here
        lemma_char_vecs = []
        for char in lemma:
            if char in self.alphabet_index:
                lemma_char_vecs.append(self.char_lookup[self.alphabet_index[char]])
            else:
                lemma_char_vecs.append(self.char_lookup[self.alphabet_index[UNK]])
        return lemma_char_vecs


    def build_encoder(self):
        # this is used in both soft and hard attention systems
        self.encoder_frnn = pc.LSTMBuilder(self.layers, self.input_dim, self.hidden_dim, self.model)
        self.encoder_rrnn = pc.LSTMBuilder(self.layers, self.input_dim, self.hidden_dim, self.model)


    def bilstm_transduce(self, lemma_char_vecs):
        # used in encoder of both soft and hard attention systems

        # BiLSTM forward pass
        s_0 = self.encoder_frnn.initial_state()
        s = s_0
        frnn_outputs = []
        for c in lemma_char_vecs:
            s = s.add_input(c)
            frnn_outputs.append(s.output())

        # BiLSTM backward pass
        s_0 = self.encoder_rrnn.initial_state()
        s = s_0
        rrnn_outputs = []
        for c in reversed(lemma_char_vecs):
            s = s.add_input(c)
            rrnn_outputs.append(s.output())

        # BiLTSM outputs
        blstm_outputs = []
        lemma_char_vecs_len = len(lemma_char_vecs)
        for i in xrange(lemma_char_vecs_len):
            blstm_outputs.append(
                pc.concatenate([frnn_outputs[i],
                                rrnn_outputs[lemma_char_vecs_len - i - 1]]))
        return blstm_outputs


    def build_decoder(self):
        # used in softmax output of the decoder
        alphabet_len = len(self.alphabet)
        self.R = self.model.add_parameters((alphabet_len, self.hidden_dim))
        self.bias = self.model.add_parameters(alphabet_len)


    def load_best_model(self):
        self.build_model()
        print 'trying to load model from: {}'.format(self.tmp_model_path)
        self.model.load(self.tmp_model_path)


    def save_pycnn_model(self):
        print 'saving to {}'.format(self.tmp_model_path)
        self.model.save(self.tmp_model_path)
        print 'saved to {}'.format(self.tmp_model_path)


    def log_to_file(self, e, avg_loss, train_accuracy, dev_accuracy):
        # if first write, add headers
        if e == 0:
            self.log_to_file('epoch', 'avg_loss', 'train_accuracy', 'dev_accuracy')

        with open(self.log_file_name, "a") as logfile:
            logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_loss, train_accuracy, dev_accuracy))


    def one_word_loss(self, sample):
        #return scalar loss for sample
        pass

    def train_model(self):
        print 'training...'

        if self.optimization == 'ADAM':
            self.trainer = pc.AdamTrainer(self.model, lam=REGULARIZATION, alpha=LEARNING_RATE,
                                     beta_1=0.9, beta_2=0.999, eps=1e-8)
        elif self.optimization == 'MOMENTUM':
            self.trainer = pc.MomentumSGDTrainer(self.model)
        elif self.optimization == 'SGD':
            self.trainer = pc.SimpleSGDTrainer(self.model)
        elif self.optimization == 'ADAGRAD':
            self.trainer = pc.AdagradTrainer(self.model)
        elif self.optimization == 'ADADELTA':
            self.trainer = pc.AdadeltaTrainer(self.model)
        else:
            self.trainer = pc.SimpleSGDTrainer(self.model)

        total_loss = 0.
        best_avg_dev_loss = 999.
        best_dev_accuracy = -1.
        best_train_accuracy = -1.
        patience = 0
        train_len = self.train_data.length
        dev_len = self.dev_data.length
        sanity_set_size = 100
        epochs_x = []
        train_loss_y = []
        dev_loss_y = []
        train_accuracy_y = []
        dev_accuracy_y = []

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=self.epochs).start()
        avg_loss = -1

        # does not change from epoch to epoch due to re-shuffling
        # TODO maybe this could simply be added as a self.dev_set
        dev_set = self.dev_data.iter()

        for e in xrange(self.epochs):

            train_set = self.train_data.iter(shuffle=True)

            # compute loss for each sample and update
            for i, sample in enumerate(train_set):
                loss = self.one_word_loss(sample)

                loss_value = loss.value()
                total_loss += loss_value
                loss.backward()
                self.trainer.update()
                if i > 0:
                    avg_loss = total_loss / (i + e * train_len)
                else:
                    avg_loss = total_loss

            if EARLY_STOPPING:

                # get train accuracy
                print 'evaluating on train...'
                train_predictions = self.predict_sequences(
                    self.train_data.lemmas[:sanity_set_size],
                    self.train_data.feat_dicts[:sanity_set_size])

                train_accuracy = self.evaluate_model(
                    train_predictions,
                    self.train_data.iter(indices=sanity_set_size),
                    print_results=False)[1]

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy

                dev_accuracy = 0.
                avg_dev_loss = 0.

                if dev_len > 0:

                    # get dev accuracy
                    dev_predictions = self.predict_sequences(
                        self.dev_data.lemmas,
                        self.dev_data.feat_dicts)

                    print 'evaluating on dev...'
                    # get dev accuracy
                    dev_accuracy = self.evaluate_model(
                        dev_predictions,
                        self.dev_data.iter(),
                        # print results every 5th epoch
                        print_results=(True if e % 5 == 0 else False))[1]

                    if dev_accuracy > best_dev_accuracy:
                        best_dev_accuracy = dev_accuracy

                        # save best model to disk
                        # TODO rename method
                        self.save_pycnn_model()
                        print 'saved new best model'
                        patience = 0
                    else:
                        patience += 1

                    # found "perfect" model
                    if dev_accuracy == 1:
                        train_progress_bar.finish()
                        if self.plot:
                            plt.cla()
                        return e

                    # get dev loss
                    total_dev_loss = 0.
                    for dev_sample in dev_set:
                        total_dev_loss += self.one_word_loss(sample).value()

                    avg_dev_loss = total_dev_loss / dev_len

                    if avg_dev_loss < best_avg_dev_loss:
                        best_avg_dev_loss = avg_dev_loss

                    print ('epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} '
                           'train accuracy: {4:.4f} best dev accuracy: {5:.4f} best train accuracy: {6:.4f} '
                           'patience = {7}').format(e, avg_loss, avg_dev_loss, dev_accuracy, train_accuracy,
                                                    best_dev_accuracy, best_train_accuracy, patience)

                    self.log_to_file(e, avg_loss, train_accuracy, dev_accuracy)

                    if patience == MAX_PATIENCE:
                        print 'out of patience after {0} epochs'.format(e)
                        # TODO: would like to return best model but pycnn has a bug with save and load. Maybe copy via code?
                        # return best_model[0]
                        train_progress_bar.finish()
                        if self.plot:
                            plt.cla()
                        return e
                else:

                    # if no dev set is present, optimize on train set
                    print ('no dev set for early stopping, running all epochs until perfectly fitting '
                           'or patience was reached on the train set')

                    if train_accuracy > best_train_accuracy:
                        best_train_accuracy = train_accuracy

                        # save best model to disk
                        self.save_pycnn_model()
                        print 'saved new best model'
                        patience = 0
                    else:
                        patience += 1

                    print ('epoch: {0} train loss: {1:.4f} train accuracy = {2:.4f} best train accuracy: {3:.4f} '
                           'patience = {4}').format(e, avg_loss, train_accuracy, best_train_accuracy, patience)

                    # found "perfect" model on train set or patience has reached
                    if train_accuracy == 1 or patience == MAX_PATIENCE:
                        train_progress_bar.finish()
                        if self.plot:
                            plt.cla()
                        return e

                # update lists for plotting
                train_accuracy_y.append(train_accuracy)
                epochs_x.append(e)
                train_loss_y.append(avg_loss)
                dev_loss_y.append(avg_dev_loss)
                dev_accuracy_y.append(dev_accuracy)

            # finished epoch
            train_progress_bar.update(e)
            if self.plot:
                with plt.style.context('fivethirtyeight'):
                    p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
                    p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
                    p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
                    p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
                    plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
                plt.savefig(result_file_png)
        train_progress_bar.finish()
        if self.plot:
            plt.cla()
        print 'finished training. average loss: {}'.format(avg_loss)
        return e


    def produce_word_from_prediction(self, predicted_sequence):
        return predicted_sequence


    def evaluate_model(self, predicted_sequences, test_data, print_results=False):
        if print_results:
            print 'evaluating model...'

        c = 0
        for sample in test_data:
            word, lemma, feat_dict = sample[:3]  # TODO Hack!!!
            joint_index = lemma + ':' + common.get_morph_string(feat_dict, self.feature_types)
            # `predicted_sequences` always returns a list, the first sequence in the list has
            # the highest probability of being the correct prediction.
            predicted_template = predicted_sequences[joint_index][0]
            predicted_word = self.produce_word_from_prediction(predicted_template)
            if predicted_word == word:
                c += 1
                sign = u'V'
            else:
                sign = u'X'
            if print_results:# and sign == 'X':
                enc_l = lemma.encode('utf8')
                enc_w = word.encode('utf8')
                enc_t = predicted_template.encode('utf8')
                enc_p = predicted_word.encode('utf8')
                print 'lemma: {}'.format(enc_l)
                print 'gold: {}'.format(enc_w)
                print 'template: {}'.format(enc_t)
                print 'prediction: {}'.format(enc_p)
                print sign

        predicted_sequences_len = len(predicted_sequences)
        accuracy = c / float(predicted_sequences_len)
        if print_results:
            print 'finished evaluating model. accuracy: {0} / {1} = {2}\n\n'.format(c, predicted_sequences_len, accuracy)

        return predicted_sequences_len, accuracy


    def evaluate_ndst(self, test_data, test=False, print_results=False):
        # TODO this completely removes the functionality of evaluating an ensemble of models
        self.load_best_model()
        predicted_sequences = self.predict_sequences(test_data.lemmas,
                                                     test_data.feat_dicts)
        # run internal evaluation
        _, accuracy = self.evaluate_model(predicted_sequences,
                                          test_data.iter(),
                                          print_results=False)

        print 'accuracy: {}'.format(accuracy)
        final_results = []
        # get predicted_sequences in the same order they appeared in the original file
        # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
        for sample in test_data.iter():
            word, lemma, feat_dict = sample[:3]  # TODO hack!!!
            #print lemma, feat_dict
            joint_index = lemma + ':' + common.get_morph_string(feat_dict, self.feature_types)
            inflection = [self.produce_word_from_prediction(p) for p in predicted_sequences[joint_index]]
            final_results.append((lemma, feat_dict, inflection))

        if test:
            path = self.test_path
            suffix = '.best.test'
        else:
            path = self.dev_path
            suffix = '.best'
        output_file_path = self.results_file_path + suffix

        common.write_results_file_and_evaluate_externally(self.hyper_params, accuracy,
                                                          self.train_path, path, output_file_path,
                                                          # nbest=True simply means inflection comes as a list
                                                          {i : v for i, v in enumerate(final_results)},
                                                          nbest=True,test=test)


    def predict_sequences(self, lemmas, feats):

        if self.beam_width > 1:
            # TODO nbest implies beam search, so
            # (eval_only)              => (no training), greedy search eval, one output
            # (eval_only +) beam_width => (no training), beam search eval, one output
            # Also, nbest implies eval_only, so
            # eval_only + nbest      => beam search eval, nbest outputs with default beam_width
            # eval_only + nbest + beam_width => beam search eval, nbest outputs with beam_width
            #
            # So, flag beam is not needed, beam_width is needed,
            # beam_width = 1 implies greedy search
            # nbest and eval_only are disjoint, eval_only is just a shorthand for nbest=1
            print "Predicting using beam search with beam width {}...".format(self.beam_width)
            if self.nbest > 1:
                print "Returning {}-best hypotheses per sample...".format(self.nbest)
        else:
            print "Predicting sequences using greedy search..."

        predictions = {}
        for i, (lemma, feat_dict) in enumerate(zip(lemmas, feats)):
            predicted_sequences = self.predict_output_sequence(lemma, feat_dict)

            # index each output by its matching inputs: lemma + features
            joint_index = lemma + ':' + common.get_morph_string(feat_dict, self.feature_types)
            predictions[joint_index] = predicted_sequences  # TODO Note that this is a list!
            #print 'Predicted smth of length: ', len(predicted_sequences)
        return predictions


    def make_hypothesis(self, seq, prob, decoder_state, *args, **kwargs):
        return Hypothesis(seq, prob, decoder_state)


    def initialize_hypothesis(self, s_0, *args, **kwargs):
        return self.make_hypothesis(seq=[BEGIN_WORD], prob=0.0, decoder_state=s_0, *args, **kwargs)


    def extend_hypothesis(self, hypothesis, new_prob, new_char, s, *args, **kwargs):
        new_prob = hypothesis.prob + np.log(new_prob)
        new_seq = hypothesis.seq + [new_char]
        return self.make_hypothesis(seq=new_seq, prob=new_prob, decoder_state=s, *args, **kwargs)


    def compute_decoder_input(self, hypothesis):
        # return decoder_input
        pass


    def run_rnn_partially(self, lemma, feats):
        pass


    def predict_output_sequence(self, lemma, feats,
                                termination_condition=MAX_PREDICTION_LEN):

        pc.renew_cg()
        R = pc.parameter(self.R)
        bias = pc.parameter(self.bias)

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
                decoder_input = self.compute_decoder_input(hypothesis,
                                                           *partially_computed_results)

                s = hypothesis.decoder_state.add_input(decoder_input)

                # compute softmax probs vector and predict with argmax
                decoder_rnn_output = s.output()
                probs = pc.softmax(R * decoder_rnn_output + bias)
                probs = probs.vec_value()

                next_char_indeces = (common.argmax(probs, n=self.beam_width) if self.beam_width > 1
                                     else [common.argmax(probs)])

                for k in next_char_indeces:
                    new_hypothesis = self.extend_hypothesis(hypothesis,
                                                            probs[k],
                                                            self.alphabet[k],
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


    def fit(self):

        self.read_in_data()

        if not self.eval_only:
            # perform model training

            # define DyNET structure
            self.build_model()
            # train model
            last_epoch = self.train_model()
            # TODO outputting epoch and writing to file was in train_model_wrapper
            # apparently meant to be launched via multiprocessing. Still needed?
            print 'stopped on epoch {}'.format(last_epoch)

            #with open(self.result_epochs, 'w') as f:
            #    f.writelines(last_epochs)
            #print 'finished training all models'
        else:
            print 'skipped training by request. evaluating best models:'

        # eval on dev
        print '=========DEV EVALUATION:========='
        self.evaluate_ndst(self.dev_data)

        if self.test_path:
            # eval on test
            print '=========TEST EVALUATION:========='
            self.evaluate_ndst(self.test_data, test=True)
