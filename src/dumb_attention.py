"""Trains and evaluates a joint-structured model for inflection generation, using the sigmorphon 2016 shared task data
files and evaluation script.

Usage:
  hard_attention.py [--dynet-mem MEM][--input=INPUT] [--hidden=HIDDEN]
  [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION]
  [--learning=LEARNING] [--plot] [--eval] [--ensemble=ENSEMBLE] TRAIN_PATH DEV_PATH TEST_PATH RESULTS_PATH
  SIGMORPHON_PATH...

Arguments:
  TRAIN_PATH    destination path
  DEV_PATH      development set path
  TEST_PATH     test path
  RESULTS_PATH  results file to be written
  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
  -h --help                     show this help message and exit
  --dynet-mem MEM                 allocates MEM bytes for (py)cnn
  --input=INPUT                 input vector dimensions
  --hidden=HIDDEN               hidden layer dimensions
  --feat-input=FEAT             feature input vector dimension
  --epochs=EPOCHS               amount of training epochs
  --layers=LAYERS               amount of layers in lstm network
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
  --reg=REGULARIZATION          regularization parameter for optimization
  --learning=LEARNING           learning rate parameter for optimization
  --plot                        draw a learning curve plot while training each model
  --eval                        run evaluation without training
  --ensemble=ENSEMBLE           ensemble model paths, separated by comma
"""

import traceback
import numpy as np
import random
import prepare_sigmorphon_data
import progressbar
import datetime
import time
import common
from matplotlib import pyplot as plt
from docopt import docopt
import dynet as pc
from collections import defaultdict
import sys

# default values
INPUT_DIM = 200
FEAT_INPUT_DIM = 20
HIDDEN_DIM = 200
EPOCHS = 1
LAYERS = 2
MAX_PREDICTION_LEN = 50
OPTIMIZATION = 'ADAM'
EARLY_STOPPING = True
MAX_PATIENCE = 100
REGULARIZATION = 0.0
LEARNING_RATE = 0.0001  # 0.1
PARALLELIZE = True

NULL = '%'
UNK = '#'
EPSILON = '*'
BEGIN_WORD = '<'
END_WORD = '>'
UNK_FEAT = '@'
STEP = '^'
ALIGN_SYMBOL = '~'


def main(train_path, dev_path, test_path, results_file_path, sigmorphon_root_dir, input_dim, hidden_dim, feat_input_dim,
         epochs, layers, optimization, regularization, learning_rate, plot, eval_only, ensemble):
    hyper_params = {'INPUT_DIM': input_dim, 'HIDDEN_DIM': hidden_dim, 'FEAT_INPUT_DIM': feat_input_dim,
                    'EPOCHS': epochs, 'LAYERS': layers, 'MAX_PREDICTION_LEN': MAX_PREDICTION_LEN,
                    'OPTIMIZATION': optimization, 'PATIENCE': MAX_PATIENCE, 'REGULARIZATION': regularization,
                    'LEARNING_RATE': learning_rate}

    print 'train path = ' + str(train_path)
    print 'dev path =' + str(dev_path)
    print 'test path =' + str(test_path)
    for param in hyper_params:
        print param + '=' + str(hyper_params[param])

    # load train and test data
    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (dev_words, dev_lemmas, dev_feat_dicts) = prepare_sigmorphon_data.load_data(dev_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

    # used for character dropout
    alphabet.append(NULL)
    alphabet.append(UNK)

    # used during decoding
    alphabet.append(EPSILON)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    # add indices to alphabet - used to indicate when copying from lemma to word
    for marker in [str(i) for i in xrange(3 * MAX_PREDICTION_LEN)]:
        alphabet.append(marker)

    # indicates the FST to step forward in the input
    alphabet.append(STEP)

    # char 2 int
    alphabet_index = dict(zip(alphabet, range(0, len(alphabet))))
    inverse_alphabet_index = {index: char for char, index in alphabet_index.items()}

    # feat 2 int
    feature_alphabet = common.get_feature_alphabet(train_feat_dicts)
    feature_alphabet.append(UNK_FEAT)
    feat_index = dict(zip(feature_alphabet, range(0, len(feature_alphabet))))

    if not eval_only:

        # align the words to the inflections, the alignment will later be used by the model
        print 'started aligning'
        train_word_pairs = zip(train_lemmas, train_words)
        dev_word_pairs = zip(dev_lemmas, dev_words)

        # TODO Dumb align added lines 123, 127
        train_aligned_pairs = common.dumb_align(train_word_pairs, ALIGN_SYMBOL)
        #train_aligned_pairs = common.mcmc_align(train_word_pairs, ALIGN_SYMBOL)

        # TODO: align together?
        dev_aligned_pairs = common.dumb_align(dev_word_pairs, ALIGN_SYMBOL)
        #dev_aligned_pairs = common.mcmc_align(dev_word_pairs, ALIGN_SYMBOL)
        print 'finished aligning'

        last_epochs = []
        trained_model, last_epoch = train_model_wrapper(input_dim, hidden_dim, layers, train_lemmas, train_feat_dicts,
                                                        train_words, dev_lemmas, dev_feat_dicts, dev_words,
                                                        alphabet, alphabet_index, inverse_alphabet_index, epochs,
                                                        optimization, results_file_path, train_aligned_pairs,
                                                        dev_aligned_pairs,
                                                        feat_index, feature_types, feat_input_dim, feature_alphabet,
                                                        plot)

        # print when did each model stop
        print 'stopped on epoch {}'.format(last_epoch)

        with open(results_file_path + '.epochs', 'w') as f:
            f.writelines(last_epochs)

        print 'finished training all models'
    else:
        print 'skipped training by request. evaluating best models:'

    # eval on dev
    print '=========DEV EVALUATION:========='
    evaluate_ndst(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet, feature_types,
                  hidden_dim, hyper_params, input_dim, inverse_alphabet_index, layers, results_file_path,
                  sigmorphon_root_dir, dev_feat_dicts, dev_lemmas, dev_path,
                  dev_words, train_path)

    # eval on test
    print '=========TEST EVALUATION:========='
    evaluate_ndst(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet, feature_types,
                  hidden_dim, hyper_params, input_dim, inverse_alphabet_index, layers, results_file_path,
                  sigmorphon_root_dir, test_feat_dicts, test_lemmas, test_path,
                  test_words, train_path)

    return


def train_model_wrapper(input_dim, hidden_dim, layers, train_lemmas, train_feat_dicts,
                        train_words, dev_lemmas, dev_feat_dicts, dev_words,
                        alphabet, alphabet_index, inverse_alphabet_index, epochs,
                        optimization, results_file_path, train_aligned_pairs, dev_aligned_pairs, feat_index,
                        feature_types, feat_input_dim, feature_alphabet, plot):
    # build model
    initial_model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim, layers,
                                                                         feature_types, feat_input_dim,
                                                                         feature_alphabet)

    # train model
    trained_model, last_epoch = train_model(initial_model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                                            train_lemmas,
                                            train_feat_dicts, train_words, dev_lemmas,
                                            dev_feat_dicts, dev_words, alphabet_index,
                                            inverse_alphabet_index,
                                            epochs, optimization, results_file_path,
                                            train_aligned_pairs, dev_aligned_pairs, feat_index, feature_types,
                                            plot)

    # evaluate last model on dev
    predicted_sequences = predict_sequences(trained_model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, alphabet_index,
                                            inverse_alphabet_index, dev_lemmas, dev_feat_dicts,
                                            feat_index,
                                            feature_types)
    if len(predicted_sequences) > 0:
        evaluate_model(predicted_sequences, dev_lemmas, dev_feat_dicts, dev_words, feature_types, print_results=False)
    else:
        print 'no examples in dev set to evaluate'

    return trained_model, last_epoch


def build_model(alphabet, input_dim, hidden_dim, layers, feature_types, feat_input_dim, feature_alphabet):
    print 'creating model...'

    model = pc.Model()

    # character embeddings
    char_lookup = model.add_lookup_parameters((len(alphabet), input_dim))

    # feature embeddings
    feat_lookup = model.add_lookup_parameters((len(feature_alphabet), feat_input_dim))

    # used in softmax output
    R = model.add_parameters((len(alphabet), hidden_dim))
    bias = model.add_parameters(len(alphabet))

    # rnn's
    encoder_frnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)
    encoder_rrnn = pc.LSTMBuilder(layers, input_dim, hidden_dim, model)

    # 2 * HIDDEN_DIM + input_dim, as it gets BLSTM[i], previous output
    concatenated_input_dim = 2 * hidden_dim + input_dim + len(feature_types) * feat_input_dim
    decoder_rnn = pc.LSTMBuilder(layers, concatenated_input_dim, hidden_dim, model)
    print 'decoder lstm dimensions are {} x {}'.format(concatenated_input_dim, hidden_dim)
    print 'finished creating model'

    return model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn


def load_best_model(alphabet, results_file_path, input_dim, hidden_dim, layers, feature_alphabet,
                    feat_input_dim, feature_types):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = build_model(alphabet, input_dim, hidden_dim,
                                                                 layers, feature_types,
                                                                 feat_input_dim,
                                                                 feature_alphabet)
    print 'trying to load model from: {}'.format(tmp_model_path)
    model.load(tmp_model_path)
    return model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn


def log_to_file(file_name, e, avg_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(file_name, 'epoch', 'avg_loss', 'train_accuracy', 'dev_accuracy')

    with open(file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_loss, train_accuracy, dev_accuracy))


def train_model(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, train_lemmas, train_feat_dicts, train_words, dev_lemmas,
                dev_feat_dicts, dev_words, alphabet_index, inverse_alphabet_index, epochs, optimization,
                results_file_path, train_aligned_pairs, dev_aligned_pairs, feat_index, feature_types,
                plot):
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
    patience = 0
    train_len = len(train_words)
    sanity_set_size = 100
    epochs_x = []
    train_loss_y = []
    dev_loss_y = []
    train_accuracy_y = []
    dev_accuracy_y = []
    e = -1

    # progress bar init
    widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()
    avg_loss = -1

    for e in xrange(epochs):

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_lemmas, train_feat_dicts, train_words, train_aligned_pairs)
        train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(train_set):
            lemma, feats, word, alignment = example
            loss = one_word_loss(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word,
                                 alphabet_index, alignment, feat_index, feature_types)
            loss_value = loss.value()
            total_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                avg_loss = total_loss / float(i + e * train_len)
            else:
                avg_loss = total_loss

        if EARLY_STOPPING:

            # get train accuracy
            print 'evaluating on train...'
            train_predictions = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, alphabet_index,
                                                  inverse_alphabet_index, train_lemmas[:sanity_set_size],
                                                  train_feat_dicts[:sanity_set_size],
                                                  feat_index,
                                                  feature_types)

            train_accuracy = evaluate_model(train_predictions, train_lemmas[:sanity_set_size],
                                            train_feat_dicts[:sanity_set_size],
                                            train_words[:sanity_set_size],
                                            feature_types, print_results=False)[1]

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            dev_accuracy = 0
            avg_dev_loss = 0

            if len(dev_lemmas) > 0:

                # get dev accuracy
                dev_predictions = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, alphabet_index,
                                                    inverse_alphabet_index, dev_lemmas, dev_feat_dicts, feat_index,
                                                    feature_types)
                print 'evaluating on dev...'
                # get dev accuracy
                dev_accuracy = evaluate_model(dev_predictions, dev_lemmas, dev_feat_dicts, dev_words, feature_types,
                                              print_results=True)[1]

                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy

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
                    total_dev_loss += one_word_loss(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, dev_lemmas[i],
                                                    dev_feat_dicts[i], dev_words[i], alphabet_index,
                                                    dev_aligned_pairs[i], feat_index, feature_types).value()

                avg_dev_loss = total_dev_loss / float(len(dev_lemmas))
                if avg_dev_loss < best_avg_dev_loss:
                    best_avg_dev_loss = avg_dev_loss

                print 'epoch: {0} train loss: {1:.4f} dev loss: {2:.4f} dev accuracy: {3:.4f} train accuracy = {4:.4f} \
 best dev accuracy {5:.4f} best train accuracy: {6:.4f} patience = {7}'.format(e, avg_loss, avg_dev_loss, dev_accuracy,
                                                                               train_accuracy, best_dev_accuracy,
                                                                               best_train_accuracy, patience)

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
            plt.savefig(results_file_path + '.png')
    train_progress_bar.finish()
    if plot:
        plt.cla()
    print 'finished training. average loss: ' + str(avg_loss)
    return model, e


def save_pycnn_model(model, results_file_path):
    tmp_model_path = results_file_path + '_bestmodel.txt'
    print 'saving to ' + tmp_model_path
    model.save(tmp_model_path)
    print 'saved to {0}'.format(tmp_model_path)


# noinspection PyPep8Naming,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def one_word_loss(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, word, alphabet_index, aligned_pair,
                  feat_index, feature_types):
    pc.renew_cg()

    # read the parameters
    # char_lookup = model["char_lookup"]
    # feat_lookup = model["feat_lookup"]
    # R = pc.parameter(model["R"])
    # bias = pc.parameter(model["bias"])
    R = pc.parameter(R)
    bias = pc.parameter(bias)

    padded_lemma = BEGIN_WORD + lemma + END_WORD

    # convert characters to matching embeddings
    lemma_char_vecs = encode_lemma(alphabet_index, char_lookup, padded_lemma)

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = encode_feats(feat_index, feat_lookup, feats, feature_types)

    feats_input = pc.concatenate(feat_vecs)

    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]
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
            loss.append(-pc.log(pc.pick(probs, alphabet_index[END_WORD])))
            continue

        # initially, if there is no prefix in the output (shouldn't delay on current input), step forward
        # TODO: check if can remove this condition entirely by adding '<' to both the aligned lemma/word
        if padded_lemma[i] == BEGIN_WORD and aligned_lemma[align_index] != ALIGN_SYMBOL:

            # perform rnn step
            s = s.add_input(decoder_input)
            decoder_rnn_output = s.output()
            probs = pc.softmax(R * decoder_rnn_output + bias)

            # compute local loss
            loss.append(-pc.log(pc.pick(probs, alphabet_index[STEP])))

            # prepare for the next iteration - "feedback"
            prev_output_vec = char_lookup[alphabet_index[STEP]]
            prev_char_vec = char_lookup[alphabet_index[EPSILON]]
            i += 1

        # if 0-to-1 or 1-to-1 alignment, compute loss for predicting the output symbol
        if aligned_word[align_index] != ALIGN_SYMBOL:
            decoder_input = pc.concatenate([prev_output_vec, blstm_outputs[i], feats_input])

            # feed new input to decoder
            s = s.add_input(decoder_input)
            decoder_rnn_output = s.output()
            probs = pc.softmax(R * decoder_rnn_output + bias)

            if aligned_word[align_index] in alphabet_index:
                current_loss = -pc.log(pc.pick(probs, alphabet_index[aligned_word[align_index]]))

                # prepare for the next iteration - "feedback"
                prev_output_vec = char_lookup[alphabet_index[aligned_word[align_index]]]
            else:
                current_loss = -pc.log(pc.pick(probs, alphabet_index[UNK]))

                # prepare for the next iteration - "feedback"
                prev_output_vec = char_lookup[alphabet_index[UNK]]
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
            loss.append(-pc.log(pc.pick(probs, alphabet_index[STEP])))

            # prepare for the next iteration - "feedback"
            prev_output_vec = char_lookup[alphabet_index[STEP]]
            i += 1

    # loss = esum(loss)
    loss = pc.average(loss)

    return loss


def encode_feats(feat_index, feat_lookup, feats, feature_types):
    feat_vecs = []
    for feat in sorted(feature_types):
        # TODO: is it OK to use same UNK for all feature types? and for unseen feats as well?
        # if this feature has a value, take it from the lookup. otherwise use UNK
        if feat in feats:
            feat_str = feat + ':' + feats[feat]
            try:
                feat_vecs.append(feat_lookup[feat_index[feat_str]])
            except KeyError:
                # handle UNK or dropout
                feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
        else:
            feat_vecs.append(feat_lookup[feat_index[UNK_FEAT]])
    return feat_vecs


# noinspection PyPep8Naming
def predict_output_sequence(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, lemma, feats, alphabet_index,
                            inverse_alphabet_index, feat_index, feature_types):
    pc.renew_cg()

    # read the parameters
    # char_lookup = model["char_lookup"]
    # feat_lookup = model["feat_lookup"]
    # R = pc.parameter(model["R"])
    # bias = pc.parameter(model["bias"])
    R = pc.parameter(R)
    bias = pc.parameter(bias)

    # convert characters to matching embeddings, if UNK handle properly
    padded_lemma = BEGIN_WORD + lemma + END_WORD
    lemma_char_vecs = encode_lemma(alphabet_index, char_lookup, padded_lemma)

    # convert features to matching embeddings, if UNK handle properly
    feat_vecs = encode_feats(feat_index, feat_lookup, feats, feature_types)

    feats_input = pc.concatenate(feat_vecs)

    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs)

    # initialize the decoder rnn
    s_0 = decoder_rnn.initial_state()
    s = s_0

    # set prev_output_vec for first lstm step as BEGIN_WORD
    prev_output_vec = char_lookup[alphabet_index[BEGIN_WORD]]

    # i is input index, j is output index
    i = 0
    num_outputs = 0
    predicted_output_sequence = []

    # run the decoder through the sequence and predict characters, twice max prediction as step outputs are added
    while num_outputs < MAX_PREDICTION_LEN * 3:

        # prepare input vector and perform LSTM step
        decoder_input = pc.concatenate([prev_output_vec,
                                        blstm_outputs[i],
                                        feats_input])

        s = s.add_input(decoder_input)

        # compute softmax probs vector and predict with argmax
        decoder_rnn_output = s.output()
        probs = pc.softmax(R * decoder_rnn_output + bias)
        probs = probs.vec_value()
        predicted_output_index = common.argmax(probs)
        predicted_output = inverse_alphabet_index[predicted_output_index]
        predicted_output_sequence.append(predicted_output)

        # check if step or char output to promote i.
        if predicted_output == STEP:
            if i < len(padded_lemma) - 1:
                i += 1

        num_outputs += 1

        # check if reached end of word
        if predicted_output_sequence[-1] == END_WORD:
            break

        # prepare for the next iteration - "feedback"
        prev_output_vec = char_lookup[predicted_output_index]

    # remove the end word symbol

    return u''.join(predicted_output_sequence[0:-1])


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
    lemma_char_vecs_len = len(lemma_char_vecs)
    for i in xrange(lemma_char_vecs_len):
        blstm_outputs.append(pc.concatenate([frnn_outputs[i], rrnn_outputs[lemma_char_vecs_len - i - 1]]))

    return blstm_outputs


def encode_lemma(alphabet_index, char_lookup, padded_lemma):
    lemma_char_vecs = []
    for char in padded_lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    return lemma_char_vecs


def predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, alphabet_index, inverse_alphabet_index, lemmas,
                      feats, feat_index, feature_types):
    predictions = {}
    for i, (lemma, feat_dict) in enumerate(zip(lemmas, feats)):
        predicted_sequence = predict_output_sequence(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn, lemma,
                                                     feat_dict, alphabet_index, inverse_alphabet_index, feat_index,
                                                     feature_types)

        # index each output by its matching inputs - lemma + features
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predictions[joint_index] = predicted_sequence

    return predictions


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def evaluate_model(predicted_sequences, lemmas, feature_dicts, words, feature_types, print_results=False):
    if print_results:
        print 'evaluating model...'

    test_data = zip(lemmas, feature_dicts, words)
    c = 0
    for i, (lemma, feat_dict, word) in enumerate(test_data):
        joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
        predicted_template = predicted_sequences[joint_index]
        predicted_word = predicted_sequences[joint_index].replace(STEP, '')
        if predicted_word == word:
            c += 1
            sign = u'V'
        else:
            sign = u'X'
        if print_results:# and sign == 'X':
            enc_l = lemma.encode('utf8')
            enc_w = word.encode('utf8')
            enc_t = ''.join([t.encode('utf8') for t in predicted_template])
            enc_p = predicted_word.encode('utf8')
            print 'lemma: {}'.format(enc_l)
            print 'gold: {}'.format(enc_w)
            print 'template: {}'.format(enc_t)
            print 'prediction: {}'.format(enc_p)
            print sign

    accuracy = float(c) / len(predicted_sequences)
    if print_results:
        print 'finished evaluating model. accuracy: ' + str(c) + '/' + str(len(predicted_sequences)) + '=' + \
              str(accuracy) + '\n\n'

    return len(predicted_sequences), accuracy


def evaluate_ndst(alphabet, alphabet_index, ensemble, feat_index, feat_input_dim, feature_alphabet, feature_types,
                  hidden_dim, hyper_params, input_dim, inverse_alphabet_index, layers, results_file_path,
                  sigmorphon_root_dir, test_feat_dicts, test_lemmas, test_path,
                  test_words, train_path, print_results=False):
    accuracies = []
    final_results = {}
    if ensemble:
        # load ensemble models
        ensemble_model_names = ensemble.split(',')
        print 'ensemble paths:\n'
        print '\n'.join(ensemble_model_names)
        ensemble_models = []
        for ens in ensemble_model_names:
            model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(
                alphabet,
                ens,
                input_dim,
                hidden_dim,
                layers,
                feature_alphabet,
                feat_input_dim,
                feature_types)

            ensemble_models.append((model, encoder_frnn, encoder_rrnn, decoder_rnn))

        # predict the entire test set with each model in the ensemble
        print 'predicting...'
        ensemble_predictions = []
        count = 0
        for em in ensemble_models:
            count += 1
            model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = em
            predicted_sequences = predict_sequences(model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn,
                                                    alphabet_index,
                                                    inverse_alphabet_index,
                                                    test_lemmas,
                                                    test_feat_dicts,
                                                    feat_index,
                                                    feature_types)
            ensemble_predictions.append(predicted_sequences)
            print 'finished to predict with ensemble: {}/{}'.format(count, len(ensemble_model_names))

        predicted_sequences = {}
        string_to_sequence = {}

        # perform voting for each test input - joint_index is a lemma+feats representation
        test_data = zip(test_lemmas, test_feat_dicts, test_words)
        for i, (lemma, feat_dict, word) in enumerate(test_data):
            joint_index = lemma + ':' + common.get_morph_string(feat_dict, feature_types)
            prediction_counter = defaultdict(int)

            # count votes
            for en in ensemble_predictions:
                prediction_str = ''.join(en[joint_index]).replace(STEP, '')
                prediction_counter[prediction_str] += 1
                string_to_sequence[prediction_str] = en[joint_index]
                if print_results:
                    print 'template: {} prediction: {}'.format(en[joint_index].encode('utf8'),
                                                               prediction_str.encode('utf8'))

            # return the most predicted output
            predicted_sequence_string = max(prediction_counter, key=prediction_counter.get)

            # hack: if chosen without majority, pick shortest prediction
            if prediction_counter[predicted_sequence_string] == 1:
                predicted_sequence_string = min(prediction_counter, key=len)

            if print_results:
                print 'chosen:{} with {} votes\n'.format(predicted_sequence_string.encode('utf8'),
                                                         prediction_counter[predicted_sequence_string])

            predicted_sequences[joint_index] = string_to_sequence[predicted_sequence_string]

            # progress indication
            sys.stdout.write("\r%d%%" % (float(i) / len(test_lemmas) * 100))
            sys.stdout.flush()
    else:
        # load best model - no ensemble
        best_model, char_lookup, feat_lookup, R, bias, encoder_frnn, encoder_rrnn, decoder_rnn = load_best_model(alphabet,
                                                                              results_file_path, input_dim,
                                                                              hidden_dim, layers,
                                                                              feature_alphabet, feat_input_dim,
                                                                              feature_types)
        try:
            predicted_sequences = predict_sequences(best_model,
                                                    char_lookup, feat_lookup, R, bias, encoder_frnn,
                                                    encoder_rrnn, decoder_rnn,
                                                    alphabet_index,
                                                    inverse_alphabet_index,
                                                    test_lemmas,
                                                    test_feat_dicts,
                                                    feat_index,
                                                    feature_types)
        except Exception as e:
            print e
            traceback.print_exc()

    # run internal evaluation
    try:
        accuracy = evaluate_model(predicted_sequences,
                                  test_lemmas,
                                  test_feat_dicts,
                                  test_words,
                                  feature_types,
                                  print_results=False)
        accuracies.append(accuracy)
    except Exception as e:
        print e
        traceback.print_exc()

    # get predicted_sequences in the same order they appeared in the original file
    # iterate through them and foreach concat morph, lemma, features in order to print later in the task format
    for i, lemma in enumerate(test_lemmas):
        joint_index = test_lemmas[i] + ':' + common.get_morph_string(test_feat_dicts[i], feature_types)
        inflection = ''.join(predicted_sequences[joint_index]).replace(STEP, '')
        final_results[i] = (test_lemmas[i], test_feat_dicts[i], inflection)

    accuracy_vals = [accuracies[i][1] for i in xrange(len(accuracies))]
    macro_avg_accuracy = sum(accuracy_vals) / len(accuracies)
    print 'macro avg accuracy: ' + str(macro_avg_accuracy)

    mic_nom = sum([accuracies[i][0] * accuracies[i][1] for i in xrange(len(accuracies))])
    mic_denom = sum([accuracies[i][0] for i in xrange(len(accuracies))])
    micro_average_accuracy = mic_nom / mic_denom
    print 'micro avg accuracy: ' + str(micro_average_accuracy)

    if 'test' in test_path:
        suffix = '.best.test'
    else:
        suffix = '.best'

    common.write_results_file_and_evaluate_externally(hyper_params, micro_average_accuracy, train_path,
                                                      test_path, results_file_path + suffix, sigmorphon_root_dir,
                                                      final_results)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['TRAIN_PATH']:
        train_path_param = arguments['TRAIN_PATH']
    else:
        train_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['DEV_PATH']:
        dev_path_param = arguments['DEV_PATH']
    else:
        dev_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-train'
    if arguments['TEST_PATH']:
        test_path_param = arguments['TEST_PATH']
    else:
        test_path_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/turkish-task1-dev'
    if arguments['RESULTS_PATH']:
        results_file_path_param = arguments['RESULTS_PATH']
    else:
        results_file_path_param = \
            '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/results_' + st + '.txt'
    if arguments['SIGMORPHON_PATH']:
        sigmorphon_root_dir_param = arguments['SIGMORPHON_PATH'][0]
    else:
        sigmorphon_root_dir_param = '/Users/roeeaharoni/research_data/sigmorphon2016-master/'
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
    if arguments['--eval']:
        eval_param = True
    else:
        eval_param = False
    if arguments['--ensemble']:
        ensemble_param = arguments['--ensemble']
    else:
        ensemble_param = False

    print arguments

    main(train_path_param, dev_path_param, test_path_param, results_file_path_param, sigmorphon_root_dir_param,
         input_dim_param,
         hidden_dim_param, feat_input_dim_param, epochs_param, layers_param, optimization_param, regularization_param,
         learning_rate_param, plot_param, eval_param, ensemble_param)


def encode_feats_and_chars(alphabet_index, char_lookup, encoder_frnn, encoder_rrnn, feat_index, feat_lookup, feats,
                           feature_types, lemma):
    feat_vecs = []

    # convert features to matching embeddings, if UNK handle properly
    for feat in sorted(feature_types):

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
    lemma_char_vecs = [char_lookup[alphabet_index[BEGIN_WORD]]]
    for char in lemma:
        try:
            lemma_char_vecs.append(char_lookup[alphabet_index[char]])
        except KeyError:
            # handle UNK character
            lemma_char_vecs.append(char_lookup[alphabet_index[UNK]])

    # add terminator symbol
    lemma_char_vecs.append(char_lookup[alphabet_index[END_WORD]])

    # create bidirectional representation
    blstm_outputs = bilstm_transduce(encoder_frnn, encoder_rrnn, lemma_char_vecs)
    return blstm_outputs, feat_vecs
