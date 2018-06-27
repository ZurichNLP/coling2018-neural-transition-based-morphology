import align
import codecs
import os
import heapq
from defaults import EVALM_PATH, RESULTS_PATH, DATA_PATH

NULL = '%'

def cluster_data_by_morph_type(feat_dicts, feature_types):
    morphs_to_indices = {}
    for i, feat_dict in enumerate(feat_dicts):
        s = get_morph_string(feat_dict, feature_types)
        if s in morphs_to_indices:
            morphs_to_indices[s].append(i)
        else:
            morphs_to_indices[s] = [i]
    return morphs_to_indices


def get_morph_string(feat_dict, feature_types):
    s = ''
    for f in sorted(feature_types):
        if f in feat_dict:
            s += f + '=' + feat_dict[f] + ':'
        else:
            s += f + '=' + NULL + ':'
    s = s[:-1]
    return s


def cluster_data_by_pos(feat_dicts):
    pos_to_indices = {}
    pos_key = 'pos'
    for i, d in enumerate(feat_dicts):
        if pos_key in d:
            s = pos_key + '=' + d[pos_key]
        else:
            s = pos_key + '=' + NULL
        if s in pos_to_indices:
            pos_to_indices[s].append(i)
        else:
            pos_to_indices[s] = [i]
    return pos_to_indices


def get_single_pseudo_cluster(feat_dicts):
    key = 'all'
    cluster_to_indices = {key:[]}
    for i, d in enumerate(feat_dicts):
        cluster_to_indices[key].append(i)
    return cluster_to_indices


def argmax(iterable, n=1):
    if n==1:
        return max(enumerate(iterable), key=lambda x: x[1])[0]
    else:
        return heapq.nlargest(n, xrange(len(iterable)), iterable.__getitem__)


def get_feature_alphabet(feat_dicts):
    feature_alphabet = []
    for f_dict in feat_dicts:
        for f in f_dict:
            feature_alphabet.append(f + ':' + f_dict[f])
    feature_alphabet = list(set(feature_alphabet))
    return feature_alphabet


def dumb_align(wordpairs, align_symbol):
    alignedpairs = []
    for idx, pair in enumerate(wordpairs):
        ins = pair[0]
        outs = pair[1]
        if len(ins) > len(outs):
            outs += align_symbol * (len(ins) - len(outs))
        elif len(outs) >= len(ins):
            ins += align_symbol * (len(outs) - len(ins))
        alignedpairs.append((ins, outs))
    return alignedpairs


def mcmc_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol)
    return a.alignedpairs


def med_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol=align_symbol, mode='med')
    return a.alignedpairs

def write_results_file_and_evaluate_externally(hyper_params, accuracy, train_path, test_path, output_file_path,
                                               final_results, nbest=False,test=False):
    if 'test' in test_path:
        output_file_path += '.test'

    if 'dev' in test_path:
        output_file_path += '.dev'

    # write hyperparams, micro + macro avg. accuracy
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        f.write('train path = ' + str(train_path) + '\n')
        f.write('test path = ' + str(test_path) + '\n')

        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')

        f.write('Prediction Accuracy = ' + str(accuracy) + '\n')

    # write predictions in sigmorphon format
    # if final results, write the special file name format
    if 'test-covered' in test_path:
        if 'task1' in test_path:
            task='1'
        if 'task2' in test_path:
            task='2'
        if 'task3' in test_path:
            task='3'

        if 'blstm' in output_file_path:
            model_dir = 'solutions/blstm'
        elif 'nfst' in output_file_path:
            model_dir = 'solutions/nfst'
        else:
            model_dir = 'solutions'

        if nbest:
            model_dir += '/nbest'

        results_prefix = '/'.join(output_file_path.split('/')[:-1])
        lang = train_path.split('/')[-1].replace('-task{0}-train'.format(task),'')
        predictions_path = '{0}/{3}/{1}-task{2}-solution'.format(results_prefix, lang, task, model_dir)
    else:
        predictions_path = output_file_path + '.predictions'
        if nbest:
            predictions_path += '.nbest'

    with codecs.open(test_path, 'r', encoding='utf8') as test_file:
        lines = test_file.readlines()

        print 'len of test file is {}'.format(len(lines))
        print 'len of predictions file is {}'.format(len(final_results))
        with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
            # Changes accomodate 2017 format
            # added tab splitting
            for i, line in enumerate(lines):
                if 'test-covered' in test_path:
                    lemma, morph = line.strip().split('\t')
                else:
                    lemma, word, morph = line.strip().split('\t')
                if i in final_results:
                    if nbest:
                        # Note that system scripts output: lemma, inflected form (=word), features (=morph)
                        for p in final_results[i][2]:
                            predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, p, morph))
                    else:
                        predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, final_results[i][2], morph))
                else:
                    # TODO: handle unseen morphs?
                    # print u'could not find prediction for {0} {1}'.format(lemma, morph)
                    predictions.write(u'{0}\t{1}\t{2}\n'.format(lemma, 'ERROR', morph))

    if not test:
        call_evalm(output_file_path, test_path, predictions_path)

    print ('wrote results to: ' + output_file_path + '\n'
           + output_file_path + '.evaluation' + '\n'
           + predictions_path)
    return


def call_evalm(output_file_path, test_path, predictions_path, evalm_path=EVALM_PATH, results_to_shell=True):
    # evaluate with sigmorphon script
    evaluation_path = output_file_path + '.evaluation'
    os.system('python ' + evalm_path + ' --gold ' + test_path + ' --guesses ' + predictions_path +
              ' > ' + evaluation_path)
    if results_to_shell:
        os.system('python ' + evalm_path + ' --gold ' + test_path + ' --guesses ' + predictions_path)


def check_path(path, arg_name, is_data_path=True):
    if not os.path.exists(path):
        prefix = DATA_PATH if is_data_path else RESULTS_PATH
        tmp = os.path.join(prefix, path)
        if is_data_path and not os.path.exists(tmp):
            print '%s incorrect: %s and %s' % (arg_name, path, tmp)
            raise ValueError
        else:
            path = tmp
    return path


def mirror_data(train_target_words, train_source_words, train_target_feat_dicts, train_source_feat_dicts):
    mirrored_train_target_words = train_target_words + train_source_words
    mirrored_train_source_words = train_source_words + train_target_words
    mirrored_train_target_feat_dicts = train_target_feat_dicts + train_source_feat_dicts
    mirrored_train_source_feat_dicts = train_source_feat_dicts + train_target_feat_dicts

    return mirrored_train_target_words, \
           mirrored_train_source_words, \
           mirrored_train_target_feat_dicts, \
           mirrored_train_source_feat_dicts
