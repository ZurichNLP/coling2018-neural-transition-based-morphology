import common
import prepare_sigmorphon_data
import task1_single_ms2s

def main():
    # train_path = '../data/heb/hebrew-task1-train'
    # dev_path = '../data/heb/hebrew-task1-dev'
    # test_path = '../data/heb/hebrew-task1-test'

    # train_path = '/Users/roeeaharoni/GitHub/sigmorphon2016/data/german-task1-train'
    # dev_path = '/Users/roeeaharoni/GitHub/sigmorphon2016/data/german-task1-dev'
    # test_path = '../biu/gold/german-task1-test'

    train_path = '/Users/roeeaharoni/GitHub/sigmorphon2016/data/finnish-task1-train'
    dev_path = '/Users/roeeaharoni/GitHub/sigmorphon2016/data/finnish-task1-dev'
    test_path = '../biu/gold/finnish-task1-test'

    (train_words, train_lemmas, train_feat_dicts) = prepare_sigmorphon_data.load_data(train_path)
    (dev_words, dev_lemmas, dev_feat_dicts) = prepare_sigmorphon_data.load_data(dev_path)
    (test_words, test_lemmas, test_feat_dicts) = prepare_sigmorphon_data.load_data(test_path)
    alphabet, feature_types = prepare_sigmorphon_data.get_alphabet(train_words, train_lemmas, train_feat_dicts)

    print 'started aligning'
    train_word_pairs = zip(train_lemmas, train_words)
    test_word_pairs = zip(test_lemmas, test_words)
    dev_word_pairs = zip(dev_lemmas, dev_words)
    align_symbol = '~'

    train_aligned_pairs = common.mcmc_align(train_word_pairs, align_symbol)

    index2template = {}
    for i, aligned_pair in enumerate(train_aligned_pairs):
        template = task1_single_ms2s.generate_template_from_alignment(aligned_pair)
        index2template[i] = template

    dev_handled = 0
    print 'now trying all templates on dev'
    for pair in dev_word_pairs:
        lemma, inflection = pair
        for template in index2template.values():
            prediction = task1_single_ms2s.instantiate_template(template, lemma)
            if prediction == inflection:
                dev_handled += 1
                break

    print "train templates handled {} examples in dev out of {}, {}%".format(dev_handled, len(dev_lemmas),
                                                                             float(dev_handled) / len(
                                                                                 dev_lemmas) * 100)

    test_handled = 0
    print 'now trying all templates on test'
    for pair in test_word_pairs:
        lemma, inflection = pair
        for template in index2template.values():
            prediction = task1_single_ms2s.instantiate_template(template, lemma)
            if prediction == inflection:
                test_handled += 1
                break

    print "train templates handled {} examples in test out of {}, {}%".format(test_handled, len(test_lemmas),
                                                                              float(test_handled)/len(test_lemmas)*100)




    # TODO: align together?
    # test_aligned_pairs = common.mcmc_align(test_word_pairs, align_symbol)

if __name__ == '__main__':
    main()