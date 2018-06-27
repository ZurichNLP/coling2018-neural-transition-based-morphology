import codecs
from collections import defaultdict
import operator


def main():

    # TODO: create alignment based templates
    # measure upper bound recall when using those templates
    # restricted decoding - choose most appropriate template using lstm based score
    # check how many predicted templates are present in the alignment based templates
    # NDST fixes - second floor, feedback
    #



    path = '/Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/heb-predictions.txt'
    templates = codecs.open(path, encoding="utf-8")
    temp2counts = defaultdict(int)
    for t in templates:
        temp2counts[t.strip()] = temp2counts[t.strip()] + 1

    sorted_dict = sorted(temp2counts.items(), key=operator.itemgetter(1), reverse=True)
    for x in sorted_dict:
        print u'{}\t{}'.format(x[0], x[1])


if __name__ == '__main__':
    main()