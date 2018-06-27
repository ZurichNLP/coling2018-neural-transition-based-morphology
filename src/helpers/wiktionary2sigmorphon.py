import codecs
import csv


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def main():
    csv2sigmorphon('/Users/roeeaharoni/research_data/morphology/MorphoCorpora/Dutch/DutchDev.txt')
    csv2sigmorphon('/Users/roeeaharoni/research_data/morphology/MorphoCorpora/Dutch/DutchTest.txt')
    csv2sigmorphon('/Users/roeeaharoni/research_data/morphology/MorphoCorpora/Dutch/DutchTrain.txt')

    csv2sigmorphon('/Users/roeeaharoni/research_data/morphology/MorphoCorpora/French/FrenchDev.txt')
    csv2sigmorphon('/Users/roeeaharoni/research_data/morphology/MorphoCorpora/French/FrenchTest.txt')
    csv2sigmorphon('/Users/roeeaharoni/research_data/morphology/MorphoCorpora/French/FrenchTrain.txt')

    return

    inflections_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/inflections_{}.csv'
    train_lemma_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_{}_train.txt'
    dev_lemma_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_{}_dev.txt'
    test_lemma_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_{}_test.txt'



    prefixes = ['de_noun', 'de_verb', 'es_verb', 'fi_nounadj', 'fi_verb']
    for p in prefixes:
        if p == 'fi_nounadj':
            base_amount = 6000
        else:
            base_amount = -1
        wiktionay2sigmorphon(dev_lemma_path.format(p),
                             inflections_path.format(p),
                             test_lemma_path.format(p),
                             train_lemma_path.format(p),
                             base_amount)


def csv2sigmorphon(inflections_path):
    suffix = '.sigmorphon_format.txt'
    output_path = inflections_path + suffix

    # open inflections file
    with codecs.open(inflections_path, encoding='utf8') as f:
        reader = unicode_csv_reader(f)
        inflections = list(reader)

    inflection_count = 0
    with codecs.open(output_path, "w", encoding='utf8') as output_file:
        for inflection_line in inflections:

            # print their inflections to file
            # from:
            # uitgeschoren, uitscheren, type = participle:tense = past
            # to:
            # uitscheren    type=participle,tense=past  uitgeschoren

            output_file.write(u'{}\t{}\t{}\n'.format(inflection_line[1],
                                                     inflection_line[2].replace(' = ', '=').replace(':',','),
                                                     inflection_line[0]))
            inflection_count += 1

    print '{} len: {}'.format(output_path, inflection_count)

def wiktionay2sigmorphon(dev_lemma_path, inflections_path, test_lemma_path, train_lemma_path, base_amount=-1):
    suffix = '.sigmorphon_format.txt'
    train_output_path = train_lemma_path + suffix
    dev_output_path = dev_lemma_path + suffix
    test_output_path = test_lemma_path + suffix
    # open train lemma file
    with codecs.open(train_lemma_path, encoding='utf8') as f:
        train_lemmas = [line.replace('\n', '') for line in f]

    # open dev lemma file
    with codecs.open(dev_lemma_path, encoding='utf8') as f:
        dev_lemmas = [line.replace('\n', '') for line in f]

    # open test lemma file
    with codecs.open(test_lemma_path, encoding='utf8') as f:
        test_lemmas = [line.replace('\n', '') for line in f]

    # open inflections file
    with codecs.open(inflections_path, encoding='utf8') as f:
        reader = unicode_csv_reader(f)
        inflections = list(reader)

    # write new files
    # train
    print_matching_inflections_to_sigmorphon_file(inflections, train_lemmas, train_output_path, base_amount)
    # dev
    print_matching_inflections_to_sigmorphon_file(inflections, dev_lemmas, dev_output_path)
    # test
    print_matching_inflections_to_sigmorphon_file(inflections, test_lemmas, test_output_path)


def print_matching_inflections_to_sigmorphon_file(inflections, base_forms, output_path, base_amount=-1):
    # sort inflections by base forms
    base2inflections = {}
    if base_amount != -1:
        base_forms = base_forms[0:base_amount]

    for inflection in inflections:
        base_form = inflection[1]
        if base_form in base2inflections:
            base2inflections[base_form].append(inflection)
        else:
            base2inflections[base_form] = []
            base2inflections[base_form].append(inflection)

    inflection_count = 0
    with codecs.open(output_path, "w", encoding='utf8') as output_file:

        # read base forms
        for base in base_forms:

            if base not in base2inflections:
                print base + 'is not found in inflection file'
                continue

            # find their inflections
            for inflection in base2inflections[base]:
                # print their inflections to file
                # from:
                # Ubungen	Ubung	case=accusative:number=plural
                # to:
                # Ubung	case=accusative,number=plural Ubungen
                output_file.write(u'{}\t{}\t{}\n'.format(inflection[1], inflection[2].replace(':', ','), inflection[0]))
                inflection_count += 1

    print '{} len: {}'.format(output_path, inflection_count)
    return


if __name__ == '__main__':
    main()
