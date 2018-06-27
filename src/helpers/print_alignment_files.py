import prepare_sigmorphon_data
import common


def main():
    #langs = ['russian', 'turkish', 'spanish', 'arabic', 'georgian', 'german', 'navajo', 'finnish']
    langs = ['arabic']
    sig_root = '/Users/roeeaharoni/GitHub/sigmorphon2016/'
    for lang in langs:
        train_path = '{0}/data/{1}-task1-train'.format(sig_root, lang)
        test_path = '{0}/data/{1}-task1-dev'.format(sig_root, lang)
        # load train and test data
        (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
        (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
        alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

        # align the words to the inflections, the alignment will later be used by the model
        print 'started aligning'
        train_word_pairs = zip(train_lemmas, train_words)
        test_word_pairs = zip(test_lemmas, test_words)
        align_symbol = '~'

        # train_aligned_pairs = dumb_align(train_word_pairs, align_symbol)
        train_aligned_pairs = common.mcmc_align(train_word_pairs, align_symbol)

        # TODO: align together?
        test_aligned_pairs = common.mcmc_align(test_word_pairs, align_symbol)
        # random.shuffle(train_aligned_pairs)
        # for p in train_aligned_pairs[:100]:
        #    generate_template(p)
        print 'finished aligning'
        for i, p in enumerate(test_aligned_pairs):
            print i
            print p[0]
            print p[1] + '\n'
    return





if __name__ == '__main__':
    main()
