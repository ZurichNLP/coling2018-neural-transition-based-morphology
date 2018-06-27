import os
import codecs

# transforms and merges rastogi celex files to sigmorphon style
def main():

    root_path = '/Users/roeeaharoni/git/neural_wfst/res/celex'
    output_dir = '/Users/roeeaharoni/git/morphological-reinflection/data/celex'

    fold_to_train_rows = {}
    fold_to_dev_rows = {}
    fold_to_test_rows = {}

    # initializations
    for type_dir in os.listdir(root_path):

        if type_dir == '.DS_Store':
            continue

        for fold_dir in os.listdir('{}/{}/0500/'.format(root_path, type_dir)):
            if fold_dir == '.DS_Store':
                continue

            fold_to_train_rows[fold_dir] = []
            fold_to_dev_rows[fold_dir] = []
            fold_to_test_rows[fold_dir] = []

    for type_dir in os.listdir(root_path):

        if type_dir == '.DS_Store':
            continue

        for fold_dir in os.listdir('{}/{}/0500/'.format(root_path, type_dir)):

            if fold_dir == '.DS_Store':
                continue



            for file_name in os.listdir('{}/{}/0500/{}'.format(root_path, type_dir, fold_dir)):

                if file_name == '.DS_Store':
                    continue

                file = codecs.open('{}/{}/0500/{}/{}'.format(root_path, type_dir, fold_dir, file_name), encoding="utf-8")
                rows = file.readlines()
                for row in rows:
                    row_elements = row.split()
                    new_row = '{}\tpos={}\t{}\n'.format(row_elements[0], type_dir, row_elements[1])
                    if 'train' in file_name:
                        fold_to_train_rows[fold_dir].append(new_row)
                    if 'dev' in file_name:
                        fold_to_dev_rows[fold_dir].append(new_row)
                    if 'test' in file_name:
                        fold_to_test_rows[fold_dir].append(new_row)

    transformations = '_'.join([x for x in os.listdir(root_path) if x != '.DS_Store'])


    for fold in fold_to_train_rows:
        path = '{}/{}_{}.train.txt'.format(output_dir, transformations, fold)
        with codecs.open(path, 'w', encoding='utf8') as sig_file:
            for row in fold_to_train_rows[fold]:
                sig_file.write(row)

    for fold in fold_to_dev_rows:
        path = '{}/{}_{}.dev.txt'.format(output_dir, transformations, fold)
        with codecs.open(path, 'w', encoding='utf8') as sig_file:
            for row in fold_to_dev_rows[fold]:
                sig_file.write(row)

    for fold in fold_to_test_rows:
        path = '{}/{}_{}.test.txt'.format(output_dir, transformations, fold)
        with codecs.open(path, 'w', encoding='utf8') as sig_file:
            for row in fold_to_test_rows[fold]:
                sig_file.write(row)


if __name__ == '__main__':
    main()
# TODO: add the transformation symbol between the words
# TODO: concat all the examples for each transformation to one file