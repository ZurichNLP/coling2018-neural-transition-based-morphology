import sys
import codecs

NULL = 'NULL'


def load_data(filename, task=1):
    """ Load data from file

    filename (str): file containing morphology reinflection data, 
                    whose structure depends on the task. 
                    for task 1, each line has
                    lemma feat1=value1,feat2=value2,feat3=value3... word
                    for task 2, each line has
                    source_feat1=value1,source_feat2=value2... source_word target_feat1=value1,target_feat2=value2... target_word
                    for task 3, each line has
                    source_word, feat1=value1,feat2=value2... target_word
    return tuple depending on the task. for task 1:
                    (words, lemmas, feat_dicts), where each element is a list
                    where each element in the list is one example
                    feat_dicts is a list of dictionaries, where each dictionary
                    is from feature name to value
                    similarly, for task 2:
                    (target_words, source_words, target_feat_dicts, source_feat_dicts)
                    and for task 3:
                    (target_words, source_words, target_feat_dicts)
    """

    print 'loading data from file:', filename
    print 'for task:', task
    #words, lemmas, feat_dicts = [], [], []
    sources, targets, target_feat_dicts = [], [], []
    if task == 2:
        source_feat_dicts = []
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            if line.strip() == '':
                # empty line marks end of lemma; ignore this for now
                continue
            splt = line.strip().split()
            if task in [1,3]:
                if 'test-covered' not in filename:
                    if len(splt) > 3:
                        # fix finnish ddn13 bad examples
                        print line.encode('utf8')
                        source = splt[0]
                        feats = [s for s in splt if '=' in s][0]
                        target = splt[-1]
                        print 'fixed: source: {} feats: {} target: {}'.format(source.encode('utf8'),
                                                                              feats.encode('utf8'),
                                                                              target.encode('utf8'))
                    else:
                        assert len(splt) == 3, 'bad line: ' + line.encode('utf8') + '\n'
                        source, feats, target = splt
                else:
                    assert len(splt) == 2, 'bad line: ' + line + '\n'
                    source, feats = splt
                    target = 'COVERED'
                sources.append(source)
                targets.append(target)
                target_feat_dicts.append(make_feat_dict(feats))
            else: # task = 2
                if not 'test-covered' in filename:
                    assert len(splt) == 4, 'bad line: ' + line + '\n'
                    source_feats, source, target_feats, target = splt
                else:
                    assert len(splt) == 3, 'bad line: ' + line + '\n'
                    source_feats, source, target_feats = splt
                    target = 'COVERED'
                sources.append(source)
                targets.append(target)
                source_feat_dicts.append(make_feat_dict(source_feats))
                target_feat_dicts.append(make_feat_dict(target_feats))

    print 'found', len(sources), 'examples'
    if task in [1,3]:
        tup = (targets, sources, target_feat_dicts)
    else:
        tup = (targets, sources, target_feat_dicts, source_feat_dicts)
    return tup


def get_alphabet(words, lemmas, feat_dicts, feat_dicts2=None):
    """
    Get alphabet from data

    words (list): list of words as strings
    lemmas (list): list of lemmas as strings
    feat_dicts (list): list of feature dictionaries, each dictionary
                       is from feature name to value
    feat_dicts2 (list): a possible second list of feature dictionaries
    return (alphabet, possible_feats): a tuple of
        alphabet (list): list of unique letters or features used
        possible_feats (list): list of possible feature names
    """

    alphabet = set()
    for word in words:
        for letter in word:
            alphabet.add(letter)
    for lemma in lemmas:
        for letter in lemma:
            alphabet.add(letter)
    possible_feats = set()
    feat_dicts_list = [feat_dicts]
    if feat_dicts2:
        feat_dicts_list.append(feat_dicts2)
    for feat_dicts in feat_dicts_list:
        for feat_dict in feat_dicts:
            for feat_key in feat_dict:
                possible_feats.add(feat_key)
                # string representing feature key+val
                feat = feat_key + '=' + feat_dict[feat_key]
                alphabet.add(feat)
                # also add null value in case we don't have it
                alphabet.add(feat_key + '=' + NULL)
    print 'alphabet size:', len(alphabet)
    print 'possible features:', possible_feats
    return list(alphabet), list(possible_feats)


def make_feat_dict(feats_str):
            
    feat_dict = {}
    for feat_key_val in feats_str.split(','):
        feat_key, feat_val = feat_key_val.split('=')
        feat_dict[feat_key] = feat_val                  
    return feat_dict


def convert_data_to_indices(words, lemmas, feat_dicts, alphabet_index, possible_feats, output_prefix):
    """ Convert data to indices

    words, lemmas, feat_dicts: as above
    alphabet_index (dict): dictionary from alphabet to index
    output_prefix (str): prefix for file names to write data as indices from alphabet. will write files for words, lemmas, and features
                         every line has one entry, a space-delimited list of indices representing letters (in words or lemmas) or features (in feat_dicts) 
    """

    print 'converting data to indices'
    word_filename = output_prefix + '.word'
    write_letters(word_filename, words, alphabet_index)
    lemma_filename = output_prefix + '.lemma'
    write_letters(lemma_filename, lemmas, alphabet_index)
    feats_filename = output_prefix + '.feats'
    write_features(feats_filename, feat_dicts, alphabet_index, possible_feats)


def write_features(filename, feat_dicts, alphabet_index, possible_feats):
    feats_file = open(filename, 'w')
    for feat_dict in feat_dicts:
        feats_to_write = []
        for possible_feat in possible_feats:
            if possible_feat in feat_dict:
                feat = possible_feat + '=' + feat_dict[possible_feat]
            else:
                feat = possible_feat + '=' + NULL
            assert feat in alphabet_index, 'feature ' + feat + ' not in alphabet' 
            feats_to_write.append(feat)
        feats_file.write(' '.join([str(alphabet_index[feat]) for feat in feats_to_write]) + '\n')
    feats_file.close()
    print 'features written to:', filename


def write_letters(filename, words, alphabet_index):
    f = open(filename, 'w')
    for word in words:
        # print word
        for letter in word:
            assert letter in alphabet_index, 'letter ' + letter + ' not in alphabet'
        f.write(' '.join([str(alphabet_index[letter]) for letter in word]) + '\n')
    f.close()
    print 'letters written to:', filename


def run(train_data_filename, test_data_filename, output_prefix, alphabet_filename, is_dev=True):

    # load train and test data
    train_words, train_lemmas, train_feat_dicts = load_data(train_data_filename)
    test_words, test_lemmas, test_feat_dicts = load_data(test_data_filename)

    # get alphabet from train data
    alphabet, possible_feats = get_alphabet(train_words, train_lemmas, train_feat_dicts)
    alphabet_index = dict(zip(alphabet, range(1,len(alphabet)+1)))
    alphabet_file = codecs.open(alphabet_filename, 'w', encoding='utf8')
    for letter in alphabet:
        alphabet_file.write(letter + '\n')
    alphabet_file.close()
    print 'alphabet written to:', alphabet_filename

    # convert data to indices
    convert_data_to_indices(train_words, train_lemmas, train_feat_dicts, alphabet_index, possible_feats, output_prefix + '.train.ind')
    test_dev_str = 'dev' if is_dev else 'test'
    convert_data_to_indices(test_words, test_lemmas, test_feat_dicts, alphabet_index, possible_feats, output_prefix + '.' + test_dev_str + '.ind')


if __name__ == '__main__':
    if len(sys.argv) == 6:
        assert sys.argv[5].lower() in ['true', 'false'], '<is_dev> must be true/false'
        is_dev = sys.argv[5].lower() == 'true'
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], is_dev)
    elif len(sys.argv) == 5:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], True)
    else:
        print 'USAGE: python ' + sys.argv[0] + ' <train file> <test/dev file> <output data prefix> <output alphabet file> [<is_dev>]'


