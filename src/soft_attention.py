"""Trains and evaluates a soft attention model for inflection generation, using the sigmorphon 2017 shared task data
files and evaluation script.

Usage:
  soft_attention.py [--dynet-mem MEM][--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION] [--learning=LEARNING] [--plot] [--override] [--eval] [--ensemble=ENSEMBLE] [--detect_feat_type] [--beam] [--beam-width=BEAM_WIDTH] [--nbest=NBEST] TRAIN_PATH DEV_PATH RESULTS_PATH [--test_path=TEST_PATH]

Arguments:
  TRAIN_PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV_PATH      development set path, possibly relative to "data/all/"
  RESULTS_PATH  results file to be written, possibly relative to "results"

Options:
  -h --help                     show this help message and exit
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
  --override                    override the existing model with the same name, if exists
  --ensemble=ENSEMBLE           ensemble model paths separated by a comma
  --eval                        run evaluation without training
  --detect_feat_type            detect feature types using knowledge from UniMorph Schema
  --test_path=TEST_PATH         test set path
  --beam                        use beam search (default is greedy search)
  --beam-width=BEAM_WIDTH       beam search width
  --nbest=NBEST                 print nbest results (the searach strategy defaults to beam)
"""

import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import os
import common
import dynet as pc

from matplotlib import pyplot as plt
from docopt import docopt
from collections import defaultdict

import feature_type_detection

# load default values for paths, NN dimensions, some training hyperparams
from defaults import (SRC_PATH, RESULTS_PATH, DATA_PATH,
                      INPUT_DIM, FEAT_INPUT_DIM, HIDDEN_DIM, LAYERS,
                      EPOCHS, OPTIMIZATION, DYNET_MEM)

# additional default values
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADADELTA'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1
PARALLELIZE = True
BEAM_WIDTH = 12

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'


def main(train_path, dev_path, test_path, results_file_path, input_dim, hidden_dim, feat_input_dim, epochs, layers, optimization, regularization, learning_rate, plot, override, eval_only, ensemble, detect_feature_type, beam, nbest, beam_width):

    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': regularization,
                    'LEARNING_RATE': learning_rate, 'BEAM_WIDTH': beam_width}

    print 'train path = ' + str(train_path)
    if test_path:
        print 'test path =' + str(test_path)
    else:
        print 'No test set.'
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load train and test data
    # This accounts for the possibility to select alternative feature dictionary construction methods
    load_data = ((lambda path:
                  prepare_sigmorphon_data.load_data(path,
                                                    feature_type_detection.make_feat_dict))
                 if detect_feature_type
                 else prepare_sigmorphon_data.load_data)
    
    (train_words, train_lemmas, train_feat_dicts) = load_data(train_path)
    (dev_words, dev_lemmas, dev_feat_dicts) = load_data(dev_path)
    if test_path:
        (test_words, test_lemmas, test_feat_dicts) = load_data(test_path)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)

    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    model_file_name = results_file_path + '_bestmodel.txt'
    if os.path.isfile(model_file_name) and not override:
        print 'loading existing model from {}'.format(model_file_name)
        model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a = load_best_model(alphabet, results_file_path, input_dim,
                                                                         hidden_dim, layers, feature_alphabet,
                                                                         feat_input_dim, feature_types)
        print 'loaded existing model successfully'
    else:
        print 'could not find existing model or explicit override was requested. starting training from scratch...'
        model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a = build_model(alphabet, input_dim, hidden_dim, layers,
                                                                     feature_types, feat_input_dim, feature_alphabet)
    if not eval_only:
        # start training
        trained_model, last_epoch, best_epoch = train_model(model, char_lookup, feat_lookup,  R, bias, encoder_frnn,
                                                            encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a,
                                                            train_lemmas, train_feat_dicts, train_words, dev_lemmas,
                                                            dev_feat_dicts, dev_words, alphabet_index,
                                                            inverse_alphabet_index, epochs, optimization,
                                                            results_file_path, feat_index, feature_types, plot, beam, nbest)
        model = trained_model
        print 'last epoch is {}'.format(last_epoch)
        print 'best epoch is {}'.format(best_epoch)
        print 'finished training'
    else:
        print 'skipped training, evaluating...'
    
    if test_path:
        print '... test set'
        main_evaluate_block(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet,
                            feature_types, hidden_dim, input_dim, inverse_alphabet_index, layers,
                            #
                            test_feat_dicts, test_lemmas, test_words,
                            #
                            model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                            W_c, W__a, U__a, v__a,
                            hyper_params, train_path, test_path, results_file_path, beam, nbest)
    else:
        print '... dev set'
        main_evaluate_block(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet,
                            feature_types, hidden_dim, input_dim, inverse_alphabet_index, layers,
                            #
                            dev_feat_dicts, dev_lemmas, dev_words,
                            #
                            model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                            W_c, W__a, U__a, v__a,
                            hyper_params, train_path, dev_path, results_file_path, beam, nbest)
    return



def main_evaluate_block(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet,
                        feature_types, hidden_dim, input_dim, inverse_alphabet_index, layers,
                        #
                        test_feat_dicts, test_lemmas, test_words,
                        #
                        model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                        W_c, W__a, U__a, v__a,
                        hyper_params, train_path, test_path, results_file_path, beam, nbest):
    if ensemble:
        predicted_sequences = predict_with_ensemble_majority(alphabet, alphabet_index, ensemble, feat_index,
                                                             feat_input_dim, feature_alphabet, feature_types,
                                                             hidden_dim, input_dim, inverse_alphabet_index, layers,
                                                             test_feat_dicts, test_lemmas, test_words, beam, nbest=0)
    else:
        predicted_sequences = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, alphabet_index,
                                                inverse_alphabet_index, test_lemmas, test_feat_dicts, feat_index,
                                                feature_types, beam, nbest)
    if len(predicted_sequences) > 0:
        # evaluate last model on test
        amount, accuracy = evaluate_model(predicted_sequences, test_lemmas, test_feat_dicts, test_words, feature_types, nbest, print_results=False)
        print 'initial eval: {}% accuracy'.format(accuracy)

        final_results = {}
        for i in xrange(len(test_lemmas)):
            joint_index = test_lemmas[i] + ':' + common.get_morph_string(test_feat_dicts[i], feature_types)
            if nbest != 0:
                inflection = [''.join(p) for p in predicted_sequences[joint_index]]
            else:
                inflection = ''.join(predicted_sequences[joint_index])
            inflection = predicted_sequences[joint_index]
            final_results[i] = (test_lemmas[i], test_feat_dicts[i], inflection)

        # evaluate best models
        if nbest != 0:
            print_nbest = True
        else:
            print_nbest = False

        common.write_results_file_and_evaluate_externally(hyper_params, accuracy, train_path, test_path,
                                                          results_file_path + '.external_eval.txt',
                                                          final_results, print_nbest)
    return



def predict_with_ensemble_majority(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet,
                                   feature_types, hidden_dim, input_dim, inverse_alphabet_index, layers,
                                   test_feat_dicts, test_lemmas, test_words, beam, nbest=0):

    ensemble_model_names = ensemble.split(',')
    print 'ensemble paths:\n'
    print '\n'.join(ensemble_model_names)
    ensemble_models = []

    # load ensemble models
    for ens in ensemble_model_names:
        model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a = load_best_model(alphabet, ens, input_dim, hidden_dim, layers,
                                                                         feature_alphabet, feat_input_dim,
                                                                         feature_types)

        ensemble_models.append((model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a))

    # predict the entire test set with each model in the ensemble
    ensemble_predictions = []
    for em in ensemble_models:
        model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a = em
        predicted_sequences = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, alphabet_index,
                                                inverse_alphabet_index, test_lemmas, test_feat_dicts, feat_index,
                                                feature_types, beam, nbest=0)

        ensemble_predictions.append(predicted_sequences)

    # perform voting for each test input - joint_index is a lemma+feats representation
    majority_predicted_sequences = {}
    string_to_template = {}
    test_data = zip(test_lemmas, test_feat_dicts, test_words)
    for i, (lemma, feat_dict, word) in enumerate(test_data):
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        prediction_counter = defaultdict(int)
        for ens in ensemble_predictions:
            prediction_str = ''.join(ens[joint_index])
            prediction_counter[prediction_str] += 1
            string_to_template[prediction_str] = ens[joint_index]
            # print 'template: {} prediction: {}'.format(''.join([e.encode('utf-8') for e in ens[joint_index]]),
            #                                            prediction_str.encode('utf-8'))

        # return the most predicted output
        majority_prediction_string = max(prediction_counter, key=prediction_counter.get)
        # print 'chosen:{} with {} votes\n'.format(majority_prediction_string.encode('utf-8'),
        #                                           prediction_counter[majority_prediction_string])
        majority_predicted_sequences[joint_index] = string_to_template[majority_prediction_string]

    return majority_predicted_sequences


def save_pycnn_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


def load_best_model(alphabet, results_file_path, input_dim, hidden_dim, layers, feature_alphabet,
                    feat_input_dim, feature_types):

    tmp_model_path = results_file_path + '_bestmodel.txt'
    model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a = build_model(alphabet, input_dim, hidden_dim, layers,
                                                                 feature_types, feat_input_dim, feature_alphabet)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a


# noinspection PyUnusedLocal
def build_model(alphabet, input_dim, hidden_dim, layers, feature_types, feat_input_dim, feature_alphabet):
    print 'creating model...'
    model = pc.Model()

    # character embeddings
    char_lookup = model.add_lookup_parameters((len(alphabet), input_dim))

    # feature embeddings
    feat_lookup = model.add_lookup_parameters((len(feature_alphabet), feat_input_dim))

    # used in softmax output
    R = model.add_parameters((len(alphabet), 3 * hidden_dim))
    bias = model.add_parameters(len(alphabet))

    # rnn's
    encoder_frnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)

    # attention MLPs - Loung-style with extra v_a from Bahdanau

    # concatenation layer for h (hidden dim), c (2 * hidden_dim)
    W_c = model.add_parameters((3 * hidden_dim, 3 * hidden_dim))

    # attention MLP's - Bahdanau-style
    # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
    W__a = model.add_parameters((hidden_dim, hidden_dim))

    # concatenation layer for h (hidden dim), c (2 * hidden_dim)
    U__a = model.add_parameters((hidden_dim, 2 * hidden_dim))

    # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
    v__a = model.add_parameters((1, hidden_dim))

    # 1 * HIDDEN_DIM - gets only the feedback input
    decoder_rnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)

    print 'finished creating model'

    return model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a


def train_model(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a,
                v__a, train_lemmas, train_feat_dicts, train_words, dev_lemmas,
                dev_feat_dicts, dev_words, alphabet_index, inverse_alphabet_index, epochs, optimization,
                results_file_path, feat_index, feature_types, plot, beam, nbest):
    print 'training...'

    np.random.seed(17)
    random.seed(17)

    if optimization == 'ADAM':
        trainer = pc.AdamTrainer(model, lam=REGULARIZATION, alpha=LEARNING_RATE, beta_1=0.9, beta_2=0.999, eps=1e-8)
    elif optimization == 'MOMENTUM':
        trainer = pc.MomentumSGDTrainer(model)
    elif optimization == 'SGD':
        trainer = pc.SimpleSGDTrainer(model)
    elif optimization == 'ADAGRAD':
        trainer = pc.AdagradTrainer(model)
    elif optimization == 'ADADELTA':
        trainer = pc.AdadeltaTrainer(model)
    else:
        trainer = pc.SimpleSGDTrainer(model)

    total_loss = 0
    best_avg_dev_loss = 999
    best_dev_accuracy = -1
    best_train_accuracy = -1
    best_dev_epoch = 0
    best_train_epoch = 0
    patience = 0
    train_len = len(train_words)
    train_sanity_set_size = 100
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_accuracy_y = []
    dev_accuracy_y = []

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
    avg_loss = -1
    e = 0

    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_lemmas, train_feat_dicts, train_words)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            lemma, feats, word = example
            loss = compute_loss(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c,
                                W__a, U__a, v__a, lemma, feats, word, alphabet_index, feat_index,
                                feature_types)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

            if i % 100 == 0 and i > 0:
                print 'went through {} examples out of {}'.format(i, train_len)

        if EARLY_STOPPING:
            print 'starting epoch evaluation'

            # get train accuracy
            print 'train sanity prediction:'
            train_predictions = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, alphabet_index,
                                                  inverse_alphabet_index, train_lemmas[:train_sanity_set_size],
                                                  train_feat_dicts[:train_sanity_set_size],
                                                  feat_index,
                                                  feature_types, beam, nbest=0)
            print 'train sanity evaluation:'
            train_accuracy = evaluate_model(train_predictions, train_lemmas[:train_sanity_set_size],
                                            train_feat_dicts[:train_sanity_set_size],
                                            train_words[:train_sanity_set_size],
                                            feature_types, nbest=0, print_results=False)[1]

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                best_train_epoch = e

            dev_accuracy = 0
            avg_dev_loss = 0

            if len(dev_lemmas) > 0:
                print 'dev prediction:'
                # get dev accuracy
                dev_predictions = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn,
                                                    encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, alphabet_index,
                                                    inverse_alphabet_index, dev_lemmas,
                                                    dev_feat_dicts, feat_index, feature_types, beam, nbest=0)
                print 'dev evaluation:'
                # get dev accuracy
                dev_accuracy = evaluate_model(dev_predictions, dev_lemmas, dev_feat_dicts, dev_words, feature_types, nbest=0, print_results=False)[1]

                if dev_accuracy >= best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy
                    best_dev_epoch = e

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                # found "perfect" model
                if dev_accuracy == 1:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

                # get dev loss
                total_dev_loss = 0
                for i in xrange(len(dev_lemmas)):
                    total_dev_loss += compute_loss(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, dev_lemmas[i],
                                                   dev_feat_dicts[i], dev_words[i], alphabet_index, feat_index,
                                                   feature_types).value()

                avg_dev_loss = total_dev_loss / float(len(dev_lemmas))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss

                print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} train accuracy = {4:.4f} \
 best dev accuracy {5:.4f} (epoch {8}) best train accuracy: {6:.4f} (epoch {9}) patience = {7}'.format(
                                                                                                e,
                                                                                                avg_loss,
                                                                                                avg_dev_loss,
                                                                                                dev_accuracy,
                                                                                                train_accuracy,
                                                                                                best_dev_accuracy,
                                                                                                best_train_accuracy,
                                                                                                patience,
                                                                                                best_dev_epoch,
                                                                                                best_train_epoch)

                log_to_file(results_file_path + '_log.txt', e, avg_loss, train_accuracy, dev_accuracy)

                if patience == MAX_PATIENCE:
                        print 'out of patience after {0} epochs'.format(str(e))
                        # TODO: would like to return best model but pycnn has a bug with save and load. Maybe copy via code?
                        # return best_model[0]
                        train_progress_bar.finish()
                        if plot:
                            plt.cla()
                        return model, e
            else:

                # if no dev set is present, optimize on train set
                print 'no dev set for early stopping, running all epochs until perfectly fitting or patience was \
                reached on the train set'

                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy

                    # save best model to disk
                    save_pycnn_model(model, results_file_path)
                    print 'saved new best model'
                    patience = 0
                else:
                    patience += 1

                print 'epoch: {0} train loss: {1:.4f} train accuracy = {2:.4f} best train accuracy: {3:.4f} \
                patience = {4}'.format(e, avg_loss, train_accuracy, best_train_accuracy, patience)

                # found "perfect" model on train set or patience has reached
                if train_accuracy == 1 or patience == MAX_PATIENCE:
                    train_progress_bar.finish()
                    if plot:
                        plt.cla()
                    return model, e

            # update lists for plotting
            train_accuracy_y.append(train_accuracy)
            epochs_x.append(e)
            train_loss_y.append(avg_loss)
            dev_loss_y.append(avg_dev_loss)
            dev_accuracy_y.append(dev_accuracy)

        # finished epoch
        train_progress_bar.update(e)

        if plot:
            with plt.style.context('fivethirtyeight'):
                p1, = plt.plot(epochs_x, dev_loss_y, label='dev loss')
                p2, = plt.plot(epochs_x, train_loss_y, label='train loss')
                p3, = plt.plot(epochs_x, dev_accuracy_y, label='dev acc.')
                p4, = plt.plot(epochs_x, train_accuracy_y, label='train acc.')
                plt.legend(loc='upper left', handles=[p1, p2, p3, p4])
            plt.savefig(results_file_path + 'plot.png')

    train_progress_bar.finish()
    if plot:
        plt.cla()
    print 'finished training. average loss: {} best epoch on dev: {} best epoch on train: {}'.format(str(avg_loss),
                                                                                                     best_dev_epoch,
                                                                                                     best_train_epoch)
    return model, e, best_train_epoch


def log_to_file(file_name, e, avg_loss, train_accuracy, dev_accuracy):

    # if first write, add headers
    if e == 0:
        log_to_file(file_name, 'epoch', 'avg_loss', 'train_accuracy', 'dev_accuracy')

    with open(file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_loss, train_accuracy, dev_accuracy))


# noinspection PyPep8Naming
def compute_loss(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, lemma, feats, word, alphabet_index, feat_index,
                 feature_types):
    pc.renew_cg()

    # read the parameters
    # char_lookup = model["char_lookup"]
    # feat_lookup = model["feat_lookup"]
    # R = pc.parameter(model["R"])
    # bias = pc.parameter(model["bias"])
    # W_c = pc.parameter(model["W_c"])
    # W__a = pc.parameter(model["W__a"])
    # U__a = pc.parameter(model["U__a"])
    # v__a = pc.parameter(model["v__a"])

    R = pc.parameter(R)
    bias = pc.parameter(bias)
    W_c = pc.parameter(W_c)
    W__a = pc.parameter(W__a)
    U__a = pc.parameter(U__a)
    v__a = pc.parameter(v__a)

    blstm_outputs = encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index,
                                           feat_lookup, feats, feature_types, lemma)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    loss = []
    padded_word = word + END_WORD

    # run the decoder through the output sequence and aggregate loss
    for i, output_char in enumerate(padded_word):

        # get current h of the decoder
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        attention_output_vector, alphas, W = attend(blstm_outputs, decoder_rnn_output, W_c, v__a, W__a, U__a)

        # compute output probabilities
        # print 'computing readout layer...'
        readout = R * attention_output_vector + bias

        if output_char in alphabet_index:
            current_loss = pc.pickneglogsoftmax(readout, alphabet_index[output_char])
        else:
            current_loss = pc.pickneglogsoftmax(readout, alphabet_index[UNK])

        # print 'computed readout layer'
        loss.append(current_loss)

        # prepare for the next iteration - "feedback"
        if output_char in alphabet_index:
            prev_output_vec = char_lookup[alphabet_index[output_char]]
        else:
            prev_output_vec = char_lookup[alphabet_index[UNK]]

    total_sequence_loss = pc.esum(loss)
    # loss = average(loss)

    return total_sequence_loss


def bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs):

    # BiLSTM forward pass
    s_0 = encoder_frnn.initial_state()
    s = s_0
    frnn_outputs = []
    for c in lemma_char_vecs:
        s = s.add_input(c)
        frnn_outputs.append(s.output())

    # BiLSTM backward pass
    s_0 = encoder_rrnn.initial_state()
    s = s_0
    rrnn_outputs = []
    for c in reversed(lemma_char_vecs):
        s = s.add_input(c)
        rrnn_outputs.append(s.output())

    # BiLTSM outputs
    blstm_outputs = []
    for i in xrange(len(lemma_char_vecs)):
        blstm_outputs.append(pc.concatenate([frnn_outputs[i], rrnn_outputs[len(lemma_char_vecs) - i - 1]]))

    return blstm_outputs

def predict_output_sequence_nbest(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, lemma, feats, alphabet_index, inverse_alphabet_index, feat_index, feature_types, nbest):
    pc.renew_cg()
    
    R = pc.parameter(R)
    bias = pc.parameter(bias)
    W_c = pc.parameter(W_c)
    W__a = pc.parameter(W__a)
    U__a = pc.parameter(U__a)
    v__a = pc.parameter(v__a)
    
    blstm_outputs = encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index, feat_lookup, feats, feature_types, lemma)
    
    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0
    
    # set prev_output_vec for first lstm step as BEGIN_WORD
    #prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    k = 0
    
    #beam = {-1: [([BEGIN_WORD], 1.0, s_0)]}
    hypos = [([BEGIN_WORD], 0.0, s_0)]
    
    #    print lemma
    # run the decoder through the sequence and predict characters
    while k < MAX_PREDICTION_LEN and len(hypos) > 0:
        
        
        # at each stage:
        # create all expansions from the previous beam:
        new_hypos = []
        for hypothesis in hypos:
            seq, hyp_prob, prefix_decoder = hypothesis
            #            print seq
            #            print hyp_prob
            last_hypo_char = seq[-1]
            #            print last_hypo_char
            
            # cant expand finished sequences
            if last_hypo_char == END_WORD:
                new_hypos.append((seq, hyp_prob, prefix_decoder))
                continue
            
            # get current h of the decoder
            prev_output_vec = char_lookup[alphabet_index[last_hypo_char]]
            s = prefix_decoder.add_input(prev_output_vec)
            decoder_rnn_output = s.output()
            
            # perform attention step
            attention_output_vector, alphas, W = attend(blstm_outputs, decoder_rnn_output, W_c, v__a, W__a, U__a)
            
            # compute output probabilities
            # print 'computing readout layer...'
            readout = R * attention_output_vector + bias
            
            # find n-best candidate output for current hypo
            probs = pc.softmax(readout)
            probs = probs.vec_value()
            #print probs
            
            next_char_indeces = common.argmax(probs, n=beam_width) if beam_width>1 else [common.argmax(probs)]
            #            print next_char_indeces
            for i in next_char_indeces:
                new_prob = hyp_prob  + np.log(probs[i])
                new_seq = list(seq)
                new_seq.append(inverse_alphabet_index[i])
                new_hypos.append((new_seq, new_prob, s))
    
        # add the expansions with the largest probability to the beam together with their score and prefix rnn state
        #print new_hypos
        new_probs = [p for (s, p, r) in new_hypos]
        argmax_indices = common.argmax(new_probs, n=beam_width) if beam_width>1 else [common.argmax(new_probs)]
        
        hypos = [new_hypos[l] for l in argmax_indices]
        k += 1

    final_probs = [p for (s, p, r) in new_hypos]
    final_indices = common.argmax(final_probs, n=nbest) if nbest>1 else [common.argmax(final_probs)]
    nbest_hypos = [u''.join(new_hypos[l][0][1:-1]) for l in final_indices]
    
    return nbest_hypos


def predict_output_sequence_beam(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, lemma, feats, alphabet_index, inverse_alphabet_index, feat_index, feature_types, nbest=0):
    pc.renew_cg()
    
    R = pc.parameter(R)
    bias = pc.parameter(bias)
    W_c = pc.parameter(W_c)
    W__a = pc.parameter(W__a)
    U__a = pc.parameter(U__a)
    v__a = pc.parameter(v__a)
    
    blstm_outputs = encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index, feat_lookup, feats, feature_types, lemma)
    
    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0
    
    # set prev_output_vec for first lstm step as BEGIN_WORD
    #prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    k = 0
    
    hypos = [([BEGIN_WORD], 0.0, s_0)]
    
    # run the decoder through the sequence and predict characters
    while k < MAX_PREDICTION_LEN and len(hypos) > 0:
        
        
        # at each stage:
        # create all expansions from the previous beam:
        new_hypos = []
        for hypothesis in hypos:
            seq, hyp_prob, prefix_decoder = hypothesis
            last_hypo_char = seq[-1]
            
            # cant expand finished sequences
            if last_hypo_char == END_WORD:
                new_hypos.append((seq, hyp_prob, prefix_decoder))
                continue
        
            # get current h of the decoder
            prev_output_vec = char_lookup[alphabet_index[last_hypo_char]]
            s = prefix_decoder.add_input(prev_output_vec)
            decoder_rnn_output = s.output()
            
            # perform attention step
            attention_output_vector, alphas, W = attend(blstm_outputs, decoder_rnn_output, W_c, v__a, W__a, U__a)
            
            # compute output probabilities
            # print 'computing readout layer...'
            readout = R * attention_output_vector + bias
            
            # find n-best candidate output for current hypo
            probs = pc.softmax(readout)
            probs = probs.vec_value()
            
            next_char_indeces = common.argmax(probs, n=beam_width) if beam_width>1 else [common.argmax(probs)]
            for i in next_char_indeces:
                new_prob = hyp_prob + np.log(probs[i])
                new_seq = list(seq)
                new_seq.append(inverse_alphabet_index[i])
                new_hypos.append((new_seq, new_prob, s))
    
        # add the expansions with the largest probability to the beam together with their score and prefix rnn state
        new_probs = [p for (s, p, r) in new_hypos]
        argmax_indices = common.argmax(new_probs, n=beam_width) if beam_width>1 else [common.argmax(new_probs)]
        
        # check if reached end of word in the best hypo - early stopping
        if new_hypos[argmax_indices[0]][0][-1] == END_WORD:
            break
        else:
            hypos = [new_hypos[l] for l in argmax_indices]
            k += 1

    predicted_sequence = new_hypos[argmax_indices[0]][0]
    
    # remove the beginning and end word symbol
    return u''.join(predicted_sequence[1:-1])

# noinspection PyPep8Naming
def predict_output_sequence(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, lemma, feats, alphabet_index,
                            inverse_alphabet_index, feat_index, feature_types, nbest=0):
    pc.renew_cg()

    # read the parameters
    # char_lookup = model["char_lookup"]
    # feat_lookup = model["feat_lookup"]
    # R = pc.parameter(model["R"])
    # bias = pc.parameter(model["bias"])
    # W_c = pc.parameter(model["W_c"])
    # W__a = pc.parameter(model["W__a"])
    # U__a = pc.parameter(model["U__a"])
    # v__a = pc.parameter(model["v__a"])

    R = pc.parameter(R)
    bias = pc.parameter(bias)
    W_c = pc.parameter(W_c)
    W__a = pc.parameter(W__a)
    U__a = pc.parameter(U__a)
    v__a = pc.parameter(v__a)

    blstm_outputs = encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index,
                                           feat_lookup, feats, feature_types, lemma)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
    i = 0
    predicted_sequence = []

    # run the decoder through the sequence and predict characters
    while i < MAX_PREDICTION_LEN:

        # get current h of the decoder
        s = s.add_input(prev_output_vec)
        decoder_rnn_output = s.output()

        # perform attention step
        attention_output_vector, alphas, W = attend(blstm_outputs, decoder_rnn_output, W_c, v__a, W__a, U__a)

        # compute output probabilities
        # print 'computing readout layer...'
        readout = R * attention_output_vector + bias

        # find best candidate output
        probs = pc.softmax(readout)
        next_char_index = common.argmax(probs.vec_value())
        predicted_sequence.append(inverse_alphabet_index[next_char_index])

        # check if reached end of word
        if predicted_sequence[-1] == END_WORD:
            break

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[next_char_index]
        i += 1

    # remove the end word symbol
    return u''.join(predicted_sequence[0:-1])

def encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index, feat_lookup, feats,
                           feature_types, lemma):

    # initialize sequence with begin symbol
    feat_vecs = [char_lookup[alphabet_index[BEGIN_WORD]]]

    # convert features to matching embeddings, if UNK handle properly
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feats:
            feat_str = feat + ':' + feats[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                print 'Bad feat: {}'.format(feat_str)
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])

    # convert characters to matching embeddings, if UNK handle properly
    lemma_char_vecs = []
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK character
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    # add feats in the beginning of the input sequence and terminator symbol
    feats_and_lemma_vecs = feat_vecs + lemma_char_vecs + [char_lookup[alphabet_index[END_WORD]]]

    # create bidirectional representation
    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, feats_and_lemma_vecs)
    return blstm_outputs


# Loung-style attention mechanism:
def attend(blstm_outputs, h_t, W_c, v_a, W__a, U__a):
    # iterate through input states to compute alphas
    # print 'computing scores...'
    # scores = [W_a * pc.concatenate([h_t, h_input]) for h_input in blstm_outputs]
    scores = [v_a * pc.tanh(W__a * h_t + U__a * h_input) for h_input in blstm_outputs]
    # print 'computed scores'
    # normalize to alphas using softmax
    # print 'computing alphas...'
    alphas = pc.softmax(pc.concatenate(scores))
    # print 'computed alphas...'
    # compute c using alphas
    # print 'computing c...'

    # import time
    # s = time.time()
    # dim = len(blstm_outputs[0].vec_value())
    # stacked_alphas = pc.concatenate_cols([alphas for j in xrange(dim)])
    # stacked_vecs = pc.concatenate_cols([h_input for h_input in blstm_outputs])
    # c = pc.esum(pc.cwise_multiply(stacked_vecs, stacked_alphas))
    # print "stack time:", time.time() - s

    # s = time.time()
    c = pc.esum([h_input * pc.pick(alphas, j) for j, h_input in enumerate(blstm_outputs)])
    # print "pick time:", time.time() - s
    # print 'computed c'
    # print 'c len is {}'.format(len(c.vec_value()))
    # compute output state h~ using c and the decoder's h (global attention variation from Loung and Manning 2015)
    # print 'computing h~...'
    h_output = pc.tanh(W_c * pc.concatenate([h_t, c]))
    # print 'len of h_output is {}'.format(len(h_output.vec_value()))
    # print 'computed h~'

    return h_output, alphas, W__a.value()


# Bahdanau style attention
def attend2(blstm_outputs, s_prev, y_feedback, v_a, W_a, U_a, U_o, V_o, C_o):

    # attention mechanism - Bahdanau style
    # iterate through input states to compute alphas
    # print 'computing scores...'

    # W_a: hidden x hidden, U_a: hidden x 2 hidden, v_a: hidden, each score: scalar
    scores = [v_a * pc.tanh(W_a * s_prev + U_a * h_j) for h_j in blstm_outputs]
    alphas = pc.softmax(pc.concatenate(scores))

    # c_i: 2 hidden
    c_i = pc.esum([h_input * pc.pick(alphas, j) for j, h_input in enumerate(blstm_outputs)])

    # U_o = 2l x hidden, V_o = 2l x input, C_o = 2l x 2 hidden
    attention_output_vector = U_o * s_prev + V_o * y_feedback + C_o * c_i

    return attention_output_vector, alphas


def predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, alphabet_index, inverse_alphabet_index, lemmas,
                      feats, feat_index, feature_types, beam=False, nbest=0):
#    print 'predicting...'
    if nbest != 0:
        print "Predicting {}-best sequences using beam search..".format(nbest)
        predict_output_sequence_template=predict_output_sequence_nbest
    elif beam == True:
        print "Predicting sequences using beam search with beam width {}".format(beam_width)
        predict_output_sequence_template=predict_output_sequence_beam
    else:
        print "Predicting sequences using greedy search.."
        predict_output_sequence_template=predict_output_sequence
    
    predictions = {}
    data_len = len(lemmas)
    for i, (lemma, feat_dict) in enumerate(zip(lemmas, feats)):
        predicted_template = predict_output_sequence_template(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, W_c, W__a, U__a, v__a, lemma,
                                                     feat_dict, alphabet_index, inverse_alphabet_index, feat_index,
                                                     feature_types,nbest)
        if i % 1000 == 0 and i > 0:
            print 'predicted {} examples out of {}'.format(i, data_len)

        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predictions[joint_index] = predicted_template

    return predictions


def evaluate_model(predicted_templates, lemmas, feature_dicts, words, feature_types, nbest=0, print_results=False):
    if print_results:
        print 'evaluating model...'

    test_data = zip(lemmas, feature_dicts, words)
    c = 0
    for i, (lemma, feat_dict, word) in enumerate(test_data):
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        if nbest != 0:
            predicted_template = predicted_templates[joint_index][0]
        else:
            predicted_template = predicted_templates[joint_index]

        predicted_word = u''.join(predicted_template)
        if i < 10:
            print predicted_word
        if predicted_word == word:
            c += 1
            sign = 'V'
        else:
            sign = 'X'
        if print_results:
            enc_l = lemma.encode('utf8')
            enc_w = word.encode('utf8')
            enc_p = predicted_word.encode('utf8')
            print 'lemma: {}'.format(enc_l)
            print 'gold: {}'.format(enc_w)
            print 'prediction: {}'.format(enc_p)
            print sign

    accuracy = float(c) / len(predicted_templates)

    if print_results:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predicted_templates)) + '=' + \
              str(accuracy) + '\n\n'

    return len(predicted_templates), accuracy


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path_param = common.check_path(arguments['TRAIN_PATH'], 'TRAIN_PATH')
    else:
        train_path_param = os.path.join(DATA_PATH, 'task1/russian-train-low')
    if arguments['DEV_PATH']:
        dev_path_param = common.check_path(arguments['DEV_PATH'], 'DEV_PATH')
    else:
        dev_path_param = os.path.join(DATA_PATH, 'task1/russian-dev')
    if arguments['--test_path']:
        test_path_param = common.check_path(arguments['--test_path'], 'test_path')
    else:
        test_path_param = None
    if arguments['RESULTS_PATH']:
        results_file_path_param = common.check_path(arguments['RESULTS_PATH'],
                                                    'RESULTS_PATH', is_data_path=False)
    else:
        results_file_path_param = os.path.join(
            RESULTS_PATH, 'SOFT/results_' + st + '.txt')
    if arguments['--input']:
        input_dim_param = int(arguments['--input'])
    else:
        input_dim_param = INPUT_DIM
    if arguments['--hidden']:
        hidden_dim_param = int(arguments['--hidden'])
    else:
        hidden_dim_param = HIDDEN_DIM
    if arguments['--feat-input']:
        feat_input_dim_param = int(arguments['--feat-input'])
    else:
        feat_input_dim_param = FEAT_INPUT_DIM
    if arguments['--epochs']:
        epochs_param = int(arguments['--epochs'])
    else:
        epochs_param = EPOCHS
    if arguments['--layers']:
        layers_param = int(arguments['--layers'])
    else:
        layers_param = LAYERS
    if arguments['--optimization']:
        optimization_param = arguments['--optimization']
    else:
        optimization_param = OPTIMIZATION
    if arguments['--reg']:
        regularization_param = float(arguments['--reg'])
    else:
        regularization_param = REGULARIZATION
    if arguments['--learning']:
        learning_rate_param = float(arguments['--learning'])
    else:
        learning_rate_param = LEARNING_RATE
    if arguments['--plot']:
        plot_param = True
    else:
        plot_param = False
    if arguments['--override']:
        override_param = True
    else:
        override_param = False
    if arguments['--eval']:
        eval_param = True
    else:
        eval_param = False
    if arguments['--ensemble']:
        ensemble_param = arguments['--ensemble']
    else:
        ensemble_param = False
    if arguments['--beam']:
        beam = True
    else:
        beam = False
    if arguments['--beam-width']:
        beam_width = int(arguments['--beam-width'])
    else:
        beam_width = BEAM_WIDTH
    if arguments['--nbest']:
        nbest = int(arguments['--nbest'])
    else:
        nbest = 0
    if arguments['--detect_feat_type']:
        detect_feature_type_param = True
    else:
        detect_feature_type_param = False

    print arguments

    main(train_path_param, dev_path_param, test_path_param, results_file_path_param, input_dim_param, hidden_dim_param, feat_input_dim_param, epochs_param, layers_param, optimization_param, regularization_param, learning_rate_param, plot_param, override_param, eval_param, ensemble_param, detect_feature_type_param, beam, nbest, beam_width)
