import prepare_sigmorphon_data
import common
import codecs

BEGIN_WORD = '<s>'
END_WORD = '</s>'
NULL = '%'

def main():
    #train_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/german-task1-train'
    #test_path = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/german-task1-dev'

    # train_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_train.txt.sigmorphon_format'
    # test_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_test.txt.sigmorphon_format'
    # dev_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_dev.txt.sigmorphon_format'

    sigmorphon_path = '/Users/roeeaharoni/GitHub/sigmorphon2016'
    output_path = '/Users/roeeaharoni/GitHub/morphological-reinflection/data'
    langs = ['russian', 'georgian', 'finnish', 'arabic', 'navajo', 'spanish', 'turkish', 'german']
    train_format = '{0}/data/{1}-task1-train'
    train_output_format = '{0}/{1}-task1-train.morphtrans.txt'
    dev_format = '{0}/data/{1}-task1-dev'
    dev_output_format = '{0}/{1}-task1-dev.morphtrans.txt'

    for lang in langs:
        train_file = train_format.format(sigmorphon_path, lang)
        train_output_file = train_output_format.format(output_path,lang)
        convert_sigmorphon_to_morphtrans(train_file, train_output_file)
        dev_file = dev_format.format(sigmorphon_path, lang)
        dev_output_file = dev_output_format.format(output_path, lang)
        convert_sigmorphon_to_morphtrans(dev_file, dev_output_file)


    # convert_sigmorphon_to_morphtrans(train_path, '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_train.txt.morphtrans_format.txt')
    # convert_sigmorphon_to_morphtrans(test_path, '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_test.txt.morphtrans_format.txt', False)
    # convert_sigmorphon_to_morphtrans(dev_path, '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_dev.txt.morphtrans_format.txt', False)

def convert_sigmorphon_to_morphtrans(sig_file, morphtrans_file, create_alphabet = True):

    (words, lemmas, feat_dicts) = prepare_sigmorphon_data.load_data(sig_file)
    alphabet, feats = prepare_sigmorphon_data.get_alphabet(words, lemmas, feat_dicts)
    alphabet.append(BEGIN_WORD)
    alphabet.append(END_WORD)

    if create_alphabet:
        with codecs.open(morphtrans_file + '.word_alphabet', "w", encoding='utf8') as alphabet_file:
            alphabet_file.write(' '.join([c for c in list(alphabet) if len(c) < 2]) + ' ' + END_WORD + ' '
                                + BEGIN_WORD)

        morph2feats = common.cluster_data_by_morph_type(feat_dicts, feats)
        with codecs.open(morphtrans_file + '.morph_alphabet', "w", encoding='utf8') as alphabet_file:
            alphabet_file.write(' '.join([key for key in morph2feats.keys()]))

    with codecs.open(morphtrans_file, "w", encoding='utf8') as output_file:
        for lemma, word, dict in zip(lemmas, words, feat_dicts):
            # <s> a b g a s k l a p p e </s>|<s> a b g a s k l a p p e </s>|case=nominative:number=singular
            output_file.write(BEGIN_WORD + ' ' + ' '.join(list(lemma)) + ' ' + END_WORD + '|' + BEGIN_WORD + ' ' +
                              ' '.join(list(word)) + ' ' + END_WORD + '|' + get_morph_string(dict, feats) + '\n')
    return

def get_morph_string(feat_dict, feats):
    s = ''
    for f in sorted(feats):
        if f in feat_dict:
            s += f + '=' + feat_dict[f] + ':'
        else:
            s += f + '=' + NULL + ':'
    return s[:-1]

if __name__ == '__main__':
    main()