import codecs
import random
from collections import defaultdict


def main():
    heb_file_path = '/Users/roeeaharoni/research_data/morphology/bgu/bgulex.utf8.hr.txt'
    sig_output_path = '/Users/roeeaharoni/GitHub/morphological-reinflection/data/heb/'
    parsed = []
    # open the original hebrew files
    d = read(codecs.open(heb_file_path, encoding="utf-8"))
    print len(d)
    words =  d.keys()
    type2counts = defaultdict(int)
    total = 0
    found_count = 0
    for word in words:
        for anal, lemma in d[word]:
            total += 1
            feat_dict = {}
            # parse word, lemma, feats (when you know how to)
            sets = anal.split(':')
            non_empty = [s for s in sets if s]

            type = non_empty[0].split('-')[0]
            type2counts[type] += 1

            found = False

            if len(non_empty) == 1:

                # handle simple verbs
                feats = non_empty[0].split('-')
                if len(feats) == 6 and non_empty[0].split('-')[0].startswith('VB'):
                    feat_dict['pos'] = feats[0]
                    feat_dict['gen'] = feats[1]
                    feat_dict['num'] = feats[2]
                    feat_dict['per'] = feats[3]
                    feat_dict['tense'] = feats[4]
                    feat_dict['binyan'] = feats[5]
                    found = True

                if non_empty[0].split('-')[0].startswith('JJT'):
                    elements = non_empty[0].split('-')
                    if len(elements) >= 3:
                        feat_dict['pos'] = elements[0]
                        feat_dict['gen'] = elements[1]
                        feat_dict['num'] = elements[2]
                    found = True

                if non_empty[0].split('-')[0].startswith('JJ'):
                    elements = non_empty[0].split('-')
                    if len(elements) >= 3:
                        feat_dict['pos'] = elements[0]
                        feat_dict['gen'] = elements[1]
                        feat_dict['num'] = elements[2]
                    found = True

                if non_empty[0].split('-')[0].startswith('NN'):
                    elements = non_empty[0].split('-')
                    if len(elements) >= 3:
                        feat_dict['pos'] = elements[0]
                        feat_dict['gen'] = elements[1]
                        feat_dict['num'] = elements[2]
                    found = True

            if len(non_empty) == 2:
                elements = non_empty[1].split('-')

                # handle definitives
                if non_empty[0].split('-')[0].startswith('DEF'):
                    found = True
                    feat_dict['def'] = 'DEF'
                    if len(elements) == 1:
                        feat_dict['pos'] = elements[0]
                    if len(elements) == 2:
                        feat_dict['pos'] = elements[0]
                        feat_dict['num'] = elements[1]
                    else:
                        if len(elements) >= 3:
                            feat_dict['pos'] = elements[0]
                            feat_dict['gen'] = elements[1]
                            feat_dict['num'] = elements[2]

                        if len(elements) >= 4:
                            feat_dict['per'] = elements[3]

                        if len(elements) >= 5:
                            feat_dict['binyan'] = elements[4]

                # handle nouns
                if non_empty[0].split('-')[0].startswith('NN'):
                    elements = non_empty[0].split('-')
                    s_elements = non_empty[1].split('-')
                    if len(elements) >= 3:
                        feat_dict['pos'] = elements[0]
                        feat_dict['gen'] = elements[1]
                        feat_dict['num'] = elements[2]

                    if len(elements) >= 4:
                        feat_dict['per'] = elements[3]

                    if len(elements) >= 5:
                        feat_dict['binyan'] = elements[4]

                    # now handle possesive
                    if len(s_elements) >= 3:
                        feat_dict['poss_gen'] = s_elements[1]
                        feat_dict['poss_num'] = s_elements[2]

                    if len(s_elements) >= 4:
                        feat_dict['poss_per'] = s_elements[3]

                    if len(s_elements) >= 5:
                        feat_dict['poss_binyan'] = s_elements[4]
                    found = True
                if non_empty[0].split('-')[0].startswith('BN'):
                    found = True
                if non_empty[0].split('-')[0].startswith('JJ'):
                    found = True
                if non_empty[0].split('-')[0].startswith('BNT'):
                    found = True
                if non_empty[0].split('-')[0].startswith('NNP'):
                    found = True
                if found == False:
                    len(feat_dict)
                    # print 'unknown type {}'.format(non_empty[0].split('-')[0])

            if len(feat_dict) == 1:
                print 'HAI'
                print word.encode("utf-8"), "\t", anal.encode("utf-8"), lemma.encode("utf-8")
            else:
                if len(feat_dict) > 0:
                    found_count += 1
                    # feat_string = ','.join([k + '=' + feat_dict[k] for k in feat_dict.keys()])
                    # print '{}\t{}\t{}'.format(lemma.encode("utf-8"), feat_string, word.encode("utf-8"))
                    parsed.append((lemma, feat_dict, word))
                else:
                    len(feat_dict)
                    # print 'NO PARSE\n'


    import operator

    sorted_dict = sorted(type2counts.items(), key=operator.itemgetter(1))
    for key in sorted_dict:
        print key

    print 'parsed {} out of {} ({:.2f}%)'.format(found_count, total, float(found_count)/total*100)

    nouns = []
    verbs = []
    adjectives = []

    parsed_type2counts = defaultdict(int)
    for p in parsed:
        lemma, feats, word = p
        if 'pos' in feats.keys():
            parsed_type2counts[feats['pos']] += 1
            if feats['pos'] == 'NN':
                nouns.append(p)
            if feats['pos'] == 'VB':
                verbs.append(p)
            if feats['pos'] == 'JJ':
                adjectives.append(p)
        else:
            print feats

    sorted_dict = sorted(parsed_type2counts.items(), key=operator.itemgetter(1))
    for key in sorted_dict:
        print key

    # from NN, VB, JJ take 2500, 7500, 2500 for train and 300, 1000, 300 for dev and 300, 1000, 300 for test
    SEED = 448
    random.seed(SEED)
    random.shuffle(verbs)
    random.shuffle(nouns)
    random.shuffle(adjectives)

    train_verbs = verbs[:7500] # 7500
    dev_verbs = verbs[7501:8500] # 1000
    test_verbs = verbs[8501:9500] # 1000

    train_nouns = nouns[:2500] # 2500
    dev_nouns = nouns[2501:2800] # 300
    test_nouns = nouns[2801:3100] # 300

    train_adjectives = adjectives[:2500] # 2500
    dev_adjectives = adjectives[2501:2800] # 300
    test_adjectives = adjectives[2801:3100] # 300

    train_data = train_nouns + train_verbs + train_adjectives
    dev_data = dev_nouns + dev_verbs + dev_adjectives
    test_data = test_nouns + test_verbs + test_adjectives

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    # write files
    write_file(train_data, sig_output_path + 'hebrew-task1-train')
    write_file(dev_data, sig_output_path + 'hebrew-task1-dev')
    write_file(test_data, sig_output_path + 'hebrew-task1-test')
    return

def write_file(train_data, path):
    with codecs.open(path, 'w', encoding='utf8') as sig_file:
        for t in train_data:
            lemma, feats, word = t
            feat_string = ','.join([k + '=' + feats[k] for k in feats.keys()])
            sig_file.write(u'{0}\t{1}\t{2}\n'.format(lemma, feat_string, word))
    print 'wrote file to: {}'.format(path)


def read(lexfile):
    d = defaultdict(list)
    for line in lexfile:
        line = line.strip().split()
        word = line[0]
        rest = iter(line[1:])
        while True:
            try:
                anal  = rest.next()
                lemma = rest.next()
                d[word].append((anal,lemma))
            except StopIteration:
                break
    return d


if __name__ == '__main__':
    main()