# this script samples a sigmorphon file into variable length files

import codecs
import random

def main():

    input_file = '../../biu/gold/finnish-task1-train'
    output_root_path = '../../data/sigmorphon_sampled'

    # open sigmorphon file
    with codecs.open(input_file, encoding='utf8') as f:
        lines = f.readlines()

    # shuffle sigmorphon file
    random.seed = 1
    random.shuffle(lines)

    # sample folds, each of 500, 1000, 3000, 5000, 7000, 9000, 12000
    folds = [500, 1000, 3000, 5000, 7000, 9000, 12000]

    prefix = input_file.split('/')[-1].split('-')[0]

    for fold in folds:
        # print folds to new files
        file_name = '{}/{}{}'.format(output_root_path, prefix, fold)
        with codecs.open(file_name, 'w', encoding='utf8') as sig_file:
            sig_file.writelines(lines[0:fold])

if __name__ == '__main__':
    main()