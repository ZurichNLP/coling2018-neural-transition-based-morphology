import sys
import codecs
import os

def main():

    # TODO: see why faruqui outputs are missing + why the features are changing order

    # train ensemble
    for i in xrange(1):

        sigmorphon_root_dir = '/Users/roeeaharoni/GitHub/sigmorphon2016'
        langs = ['georgian', 'finnish', 'arabic', 'navajo', 'spanish', 'turkish', 'german', 'russian']
        train_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/data/{0}-task1-train.morphtrans.txt'
        dev_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/data/{0}-task1-dev.morphtrans.txt'

        for lang in langs:
            train_file = train_format.format(lang)
            dev_file = dev_format.format(lang)
            train_alphabet_file = train_file + '.word_alphabet'
            train_morph_file = train_file + '.morph_alphabet'
            model_file = '/Users/roeeaharoni/research_data/morphology/morph_trans_models/model_{0}.txt'.format(
                lang)

            eval_file = "/Users/roeeaharoni/research_data/morphology/morph_trans_models/eval_model_{0}.txt".format(
                lang)

            sig_file = "/Users/roeeaharoni/research_data/morphology/morph_trans_models/sig_output_{0}.txt".format(
                lang)

            convert_morphtrans_predictions_to_sigmorphon_predictions(eval_file, sig_file)
            sig_dev_file_path = '{0}/data/{1}-task1-dev'.format(sigmorphon_root_dir, lang)
            normalized_sig_dev_file_path = "/Users/roeeaharoni/research_data/morphology/morph_trans_models/sig_dev_sorted_{0}.txt".format(
                lang)

            sort_feats_in_sig_file(sig_dev_file_path, normalized_sig_dev_file_path)

            # print train_file
            # print dev_file
            # print train_alphabet_file
            # print train_morph_file
            # print model_file
            # print eval_file
            print sig_file
            print sig_dev_file_path

            os.system(
                'python ' + sigmorphon_root_dir + '/src/evalm.py --gold '
                + normalized_sig_dev_file_path + ' --guesses ' + sig_file)
            continue

            os.system("/Users/roeeaharoni/GitHub/morph-trans/bin/train-sep-morph --cnn-mem 9064\
            {0} {1} {2} {3} \
            100 30 1e-5 2 {4}".format(
                train_alphabet_file,
                train_morph_file,
                train_file,
                dev_file,
                model_file))

            # test ensemble
            os.system("~/GitHub/morph-trans/bin/eval-ensemble-sep-morph --cnn-mem 9064\
    {0} {1} {2} {3} > {4}".format(
                train_alphabet_file,
                train_morph_file,
                dev_file,
                model_file,
                eval_file))


def sort_feats_in_sig_file(input, output):
    with codecs.open(input, "r", encoding='utf8') as sig_file:
        lines = sig_file.readlines()
        with codecs.open(output, "w", encoding='utf8') as output_file:
            for l in lines:
                lemma, feats, inflection = l.split('\t')
                sorted_feats = ','.join(sorted(feats.split(',')))
                output_file.write(u'{0}\t{1}\t{2}'.format(lemma, sorted_feats, inflection))


def convert_morphtrans_predictions_to_sigmorphon_predictions(morphtrans_output_path, sigmorphon_output_path):
    with codecs.open(morphtrans_output_path, "r", encoding='utf8') as morphtrans_file:
        morphtrans_lines = morphtrans_file.readlines()
        with codecs.open(sigmorphon_output_path, "w", encoding='utf8') as sigmorphon_file:
            for l in morphtrans_lines:
                if l.startswith('GOLD'):
                    continue
                else:
                    split_elements = l.split('|')
                    lemma = ''.join(split_elements[0].replace('<s>','').replace('</s>','').replace('PRED:','').split(' ')).strip()
                    prediction = ''.join(split_elements[1].replace('<s>','').replace('</s>','').split(' ')).strip()
                    features = ','.join(sorted([f for f in split_elements[2].replace(':',',').strip().split(',') if not '%' in f]))
                    sigmorphon_file.write(u'{0}\t{1}\t{2}\n'.format(lemma, features, prediction))




if __name__ == '__main__':
    main()