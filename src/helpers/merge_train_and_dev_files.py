import codecs

def main():
    data_path = '/Users/roeeaharoni/GitHub/sigmorphon2016/data'
    langs = ['russian', 'georgian', 'finnish', 'arabic', 'navajo', 'spanish', 'turkish', 'german',
             'hungarian', 'maltese']
    train_file_format = '{}/{}-task{}-train'
    dev_file_format = '{}/{}-task{}-dev'
    merged_file_format = '../../data/sigmorphon_train_dev_merged/{}-task{}-merged'

    # unify train and dev files into one file for all langs and tasks
    for lang in langs:
        for task in [1,2,3]:
            train_file_path = train_file_format.format(data_path, lang, task)
            dev_file_path = dev_file_format.format(data_path, lang, task)
            merged_file_path = merged_file_format.format(lang, task)

            with codecs.open(train_file_path, 'rb', encoding='utf-8') as train_file:
                train_lines = train_file.readlines()

            with codecs.open(dev_file_path, 'rb', encoding='utf-8') as dev_file:
                dev_lines = dev_file.readlines()

            with codecs.open(merged_file_path, 'w', encoding='utf-8') as merged_file:
                merged_file.writelines(train_lines + dev_lines)

    return

if __name__ == '__main__':
    main()