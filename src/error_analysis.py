"""Runs a script on selected languages and settings

    Usage:
    error_analysis.py [--langs=LANGS] [--prefix=PREFIX] [--regimes=REGIME] [--data_path=DATA_PATH] [--results_path=RESULTS_PATH] [--task=TASK] [--compare] [--prefix_2=PREFIX] [--results_path_2=RESULTS_PATH] [--format_2016] [--format_2016_2]

    Options:
    -h --help                     show this help message and exit
    --langs=LANGS                 languages separated by comma
    --prefix=PREFIX               the output files prefix
    --regimes=REGIME              low,medium,high
    --data_path=DATA_PATH         sigmorphon root containing data (but not src/evalm.py)
    --results_path=RESULTS_PATH   results file to be written
    --task=TASK                   the current task to train
    --compare                     compare error analysis (from different paths with prefixes)
    --results_path_2=RESULTS_PATH results file to be written
    --prefix_2=PREFIX             the current task to train
    --format_2016                 the prediction file is in 2016 format
    --format_2016_2               the prediction file for comparison is in 2016 format
    """

from __future__ import division
import os
import time
import datetime
import docopt
import codecs

# load default values for paths, NN dimensions, some training hyperparams
from defaults import (RESULTS_PATH, DATA_PATH)

LANGS = ['albanian', 'arabic', 'armenian', 'bulgarian', 'catalan', 'czech', 'danish', 'dutch', 'english',
         'faroese', 'finnish', 'french', 'georgian', 'german', 'hebrew', 'hindi', 'hungarian', 'icelandic',
         'italian', 'latvian', 'lower-sorbian', 'macedonian', 'navajo', 'northern-sami', 'norwegian-nynorsk',
         'persian', 'polish', 'portuguese', 'quechua', 'russian', 'scottish-gaelic', 'serbo-croatian', 'slovak',
         'slovene', 'spanish', 'swedish', 'turkish', 'ukrainian', 'urdu', 'welsh']
REGIME = ['low', 'medium'] #['low', 'medium', 'high']

def main(results_dir, sigmorphon_root_dir, langs, prefix, regimes, task, format_2016):

    if prefix == "BASELINE":
        predicted_file_format = '{0}/{1}/{2}-{3}-out'
    else:
        #predicted_file_format = '{0}/{1}/{2}-{3}.best.dev.predictions.nbest'
        predicted_file_format = '{0}/{1}/{2}-{3}-out-dev'
    gold_file_format = '{0}/task{1}/{2}-dev'
    output_file_format = '{0}/{1}/{2}_{3}.dev.error_analysis.txt'
    train_file_format = '{0}/task{1}/{2}-train-{3}'

    for lang in langs:
        for regime in regimes:

            predicted_file = predicted_file_format.format(results_dir,prefix,lang,regime)
            gold_file = gold_file_format.format(sigmorphon_root_dir,task,lang)
            output_file = output_file_format.format(results_dir,prefix,lang,regime)
            train_file = train_file_format.format(sigmorphon_root_dir,task,lang,regime)

            evaluate(predicted_file, gold_file, output_file,train_file,format_2016)
            print 'created error analysis for {0} in {1} regime in: {2}'.format(lang, regime, output_file)


def evaluate(predicted_file, gold_file, output_file, train_file, format_2016):
    morph2results = {}
    seen_morphs = {}
    predicted = open(predicted_file)
    gold = open(gold_file)
    train = open(train_file)
    output = open(output_file, 'w')

    #extract seen morphemes
    for i,line in enumerate(train):
        [lemma, inflection, morph] = line.strip().split('\t')
        if morph not in seen_morphs:
            seen_morphs[morph] = 1


    #loop trough predicton and gold line by line and collect mistakes
    count = 0
    count_seen = 0
    errors_count = 0
    for i, line in enumerate(zip(predicted,gold)):
        count += 1

        try:
            if not format_2016: #fix old format in our current 2017 files
                [pred_lemma, predicted_inflection, pred_morph] = line[0].strip().split('\t')
            else:
                [pred_lemma, pred_morph, predicted_inflection] = line[0].strip().split('\t')
        except:
            print line[0].strip()
            print i
            print line[0].strip().split('\t')

        [lemma, gold_inflection, morph] = line[1].strip().split('\t')

        if pred_lemma != lemma:
            print 'mismatch in index' + str(i)
            return
        if pred_morph in seen_morphs.keys():
            count_seen += 1

        if predicted_inflection == gold_inflection:
            mark = 'V'
        else:
            mark = 'X'
            errors_count += 1
            line_format = "lemma:{0}\tgold:{1}\tpredicted:{2}\n"
            output_line = line_format.format(lemma, gold_inflection, predicted_inflection)
            if morph in morph2results:
                morph2results[morph].append(output_line)
            else:
                morph2results[morph] = [output_line]
        # output.write(output_line)

    morphs_result_seen = [k for k in morph2results.keys() if k in seen_morphs.keys()]
    morphs_result_unseen = [k for k in morph2results.keys() if k not in seen_morphs.keys()]

    output.write('TOTAL PREDICTIONS : {}\n'.format(count))
    output.write('TOTAL PREDICTIONS WITH SEEN TAGS COMBINATIONS : {:.2f}%\n'.format(count_seen/count*100))
    output.write('TOTAL ERRORS : {0} ({1:.2f}%)\n'.format(errors_count,errors_count/count*100))

    if count_seen !=0:
        errors_seen = sum(len(morph2results[m]) for m in morphs_result_seen)
        output.write('TOTAL ERRORS WITH SEEN TAGS COMBINATIONS: {0} ({1:.2f}%)\n'.format(errors_seen, errors_seen/count_seen*100))
    if count-count_seen !=0:
        errors_unseen = sum(len(morph2results[m]) for m in morphs_result_unseen)
        output.write('TOTAL ERRORS WITH UNSEEN TAGS COMBINATIONS : {0} ({1:.2f}%)\n'.format(errors_unseen,errors_unseen/(count-count_seen)*100))

    for morph in sorted(morphs_result_seen):
        output.write('\n\nSEEN TAG: {}\n'.format(morph))
        for line in morph2results[morph]:
            output.write(line)
            output.write('\n')

    for morph in sorted(morphs_result_unseen):
        output.write('\n\nUNSEEN TAG: {}\n'.format(morph))
        for line in morph2results[morph]:
            output.write(line)
            output.write('\n')

def compare_error_analysis(results_dir, sigmorphon_root_dir, langs, prefix, regimes, task, results_dir_2, prefix_2, format_2016, format_2016_2):


    predicted_file_format_baseline = '{0}/{1}/{2}-{3}-out'
    #predicted_file_format = '{0}/{1}/{2}_{3}.best.dev.predictions'
    predicted_file_format = '{0}/{1}/{2}_{3}-out-dev'
    gold_file_format = '{0}/task{1}/{2}-dev'
    output_file_format = '{0}/{1}/{2}_{3}.dev.error_comparison_to_{4}.txt'
    train_file_format = '{0}/task{1}/{2}-train-{3}'

    for lang in langs:
        for regime in regimes:

            if prefix == "BASELINE":
                predicted_file = predicted_file_format_baseline.format(results_dir,prefix,lang,regime)
            else:
                predicted_file = predicted_file_format.format(results_dir,prefix,lang,regime)
            if prefix_2 == "BASELINE":
                predicted_file_2 = predicted_file_format_baseline.format(results_dir_2,prefix_2,lang,regime)
            else:
                predicted_file_2 = predicted_file_format.format(results_dir_2,prefix_2,lang,regime)
            gold_file = gold_file_format.format(sigmorphon_root_dir,task,lang)
            output_file = output_file_format.format(results_dir,prefix,lang,regime,prefix_2)
            train_file = train_file_format.format(sigmorphon_root_dir,task,lang,regime)

            compare(predicted_file, predicted_file_2, gold_file, output_file, train_file, format_2016, format_2016_2, prefix, prefix_2)
            print 'compared error analysis for {0} in {1} regime from {2} to {3}'.format(lang, regime, prefix, prefix_2)


def compare(predicted_file, predicted_file_2, gold_file, output_file, train_file, format_2016, format_2016_2, prefix, prefix_2):

    errors_both = {}
    errors_1 = {}
    errors_2 = {}
    seen_morphs = {}

    predicted = open(predicted_file)
    predicted_2 = open(predicted_file_2)
    gold = open(gold_file)
    train = open(train_file)
    output = open(output_file, 'w')

    #extract seen morphemes
    for i,line in enumerate(train):
        [lemma, inflection, morph] = line.strip().split('\t')
        if morph not in seen_morphs:
            seen_morphs[morph] = 1


    #loop trough predicton and gold line by line and collect mistakes
    count = 0
    errors_count_both = 0
    errors_count_1_only = 0
    errors_count_2_only = 0
    for i, line in enumerate(zip(predicted, predicted_2, gold)):
        count += 1

        try:
            if not format_2016: #fix old format in our current 2017 files
                [pred_lemma, predicted_inflection, pred_morph] = line[0].strip().split('\t')
            else:
                [pred_lemma, pred_morph, predicted_inflection] = line[0].strip().split('\t')
        except:
            print line[0].strip()
            print i
            print line[0].strip().split('\t')

        try:
            if not format_2016_2: #fix old format in our current 2017 files
                [pred_lemma_2, predicted_inflection_2, pred_morph_2] = line[1].strip().split('\t')
            else:
                 [pred_lemma_2, pred_morph_2, predicted_inflection_2] = line[1].strip().split('\t')
        except:
            print line[1].strip()
            print i
            print line[1].strip().split('\t')


        [lemma, gold_inflection, morph] = line[2].strip().split('\t')

        if pred_lemma != lemma:
            print 'mismatch in index' + str(i)
            return

        if predicted_inflection != gold_inflection and predicted_inflection_2 != gold_inflection:
            errors_count_both += 1
            line_format = "lemma:{0}\tgold:{1}\t{2}:{3}\t{4}:{5}\n"
            output_line = line_format.format(lemma, gold_inflection, prefix, predicted_inflection, prefix_2, predicted_inflection_2)
            if morph in errors_both:
                errors_both[morph].append(output_line)
            else:
                errors_both[morph] = [output_line]

        elif predicted_inflection != gold_inflection:
            errors_count_1_only += 1
            line_format = "lemma:{0}\tgold:{1}\t{2}:{3}\n"
            output_line = line_format.format(lemma, gold_inflection, prefix, predicted_inflection)
            if morph in errors_1:
                errors_1[morph].append(output_line)
            else:
                errors_1[morph] = [output_line]

        elif predicted_inflection_2 != gold_inflection:
            errors_count_2_only += 1
            line_format = "lemma:{0}\tgold:{1}\t{2}:{3}\n"
            output_line = line_format.format(lemma, gold_inflection, prefix_2, predicted_inflection_2)
            if morph in errors_2:
                errors_2[morph].append(output_line)
            else:
                errors_2[morph] = [output_line]


    morphs_result_seen = [k for k in errors_both.keys() if k in seen_morphs.keys()]
    morphs_result_unseen = [k for k in errors_both.keys() if k not in seen_morphs.keys()]

    output.write('TOTAL PREDICTIONS : {}\n'.format(count))
    output.write('TOTAL ERRORS IN BOTH : {0} ({1:.2f}%)\n'.format(errors_count_both,errors_count_both/count*100))
    output.write('TOTAL ERRORS IN BOTH WITH SEEN TAGS COMBINATIONS: {}\n'.format(sum(len(errors_both[m]) for m in morphs_result_seen)))
    output.write('TOTAL ERRORS IN BOTH WITH UNSEEN TAGS COMBINATIONS : {}\n'.format(sum(len(errors_both[m]) for m in morphs_result_unseen)))

    morphs_result_seen_1 = [k for k in errors_1.keys() if k in seen_morphs.keys()]
    morphs_result_unseen_1 = [k for k in errors_1.keys() if k not in seen_morphs.keys()]

    output.write('TOTAL ERRORS IN {0} : {1} ({2:.2f}%)\n'.format(prefix,errors_count_1_only,errors_count_1_only/count*100))
    output.write('TOTAL ERRORS IN {} WITH SEEN TAGS COMBINATIONS: {}\n'.format(prefix,sum(len(errors_1[m]) for m in morphs_result_seen_1)))
    output.write('TOTAL ERRORS IN {} WITH UNSEEN TAGS COMBINATIONS : {}\n'.format(prefix,sum(len(errors_1[m]) for m in morphs_result_unseen_1)))

    morphs_result_seen_2 = [k for k in errors_2.keys() if k in seen_morphs.keys()]
    morphs_result_unseen_2 = [k for k in errors_2.keys() if k not in seen_morphs.keys()]


    output.write('TOTAL ERRORS IN {0} : {1} ({2:.2f}%)\n'.format(prefix_2, errors_count_2_only,errors_count_2_only/count*100))
    output.write('TOTAL ERRORS IN {} WITH SEEN TAGS COMBINATIONS: {}\n'.format(prefix_2,sum(len(errors_2[m]) for m in morphs_result_seen_2)))
    output.write('TOTAL ERRORS IN {} WITH UNSEEN TAGS COMBINATIONS : {}\n'.format(prefix_2,sum(len(errors_2[m]) for m in morphs_result_unseen_2)))


    for morph in morphs_result_seen:
        output.write('\n\nSEEN TAG (both): {}\n'.format(morph))
        for line in sorted(errors_both[morph]):
            output.write(line)
            output.write('\n')

    for morph in morphs_result_unseen:
        output.write('\n\nUNSEEN TAG (both): {}\n'.format(morph))
        for line in sorted(errors_both[morph]):
            output.write(line)
            output.write('\n')

    for morph in morphs_result_seen_1:
        output.write('\n\nSEEN TAG ({}): {}\n'.format(prefix,morph))
        for line in sorted(errors_1[morph]):
            output.write(line)
            output.write('\n')

    for morph in morphs_result_unseen_1:
        output.write('\n\nUNSEEN TAG ({}): {}\n'.format(prefix,morph))
        for line in sorted(errors_1[morph]):
            output.write(line)
            output.write('\n')

    for morph in morphs_result_seen_2:
        output.write('\n\nSEEN TAG ({}): ({})\n'.format(prefix_2,morph))
        for line in sorted(errors_2[morph]):
            output.write(line)
            output.write('\n')

    for morph in morphs_result_unseen_2:
        output.write('\n\nUNSEEN TAG ({}): ({})\n'.format(prefix_2,morph))
        for line in sorted(errors_2[morph]):
            output.write(line)
            output.write('\n')

    return

def compare_error_analysis_old(results_dir, sigmorphon_root_dir, langs, prefix, regimes, task, results_dir_2, prefix_2):

    for lang in langs:
        for regime in regimes:
            error_file_1 = '{0}/{1}/{2}_{3}.dev.error_analysis.txt'.format(results_dir,prefix,lang,regime)
            error_file_2 = '{0}/{1}/{2}_{3}.dev.error_analysis.txt'.format(results_dir_2,prefix_2,lang,regime)
            output_file = '{0}/{1}/{2}_{3}.dev.error_comparison_to_{4}.txt'.format(results_dir,prefix,lang,regime,prefix_2)
            print 'compare error analysis for {0} in {1} regime from {2} to {3}'.format(lang, regime, prefix, prefix_2)

            errors_1 = {}
            errors_2 = {}
            with open(error_file_1) as ef1:
                #ef1_lines = ef1.readlines()
                for i, line in enumerate(ef1):
                    if line not in ['#################################\n','\n']:
                        [morph, lemma, gold_inflection, predicted_inflection, mark] = line.strip().split('\t')
                        errors_1["\t".join([morph, lemma, gold_inflection])] = predicted_inflection

            with open(error_file_2) as ef2:
                #ef2_lines = ef2.readlines()
                for i, line in enumerate(ef2):
                    if line not in ['#################################\n','\n']:
                        [morph, lemma, gold_inflection, predicted_inflection, mark] = line.strip().split('\t')
                        errors_2["\t".join([morph, lemma, gold_inflection])] = predicted_inflection


            print len(set(errors_1.keys()) & set(errors_2.keys()))
            errors_both={k:prefix + ":" + errors_1[k] + "\t" + prefix_2 + ":" + errors_2[k] for k in set(errors_1.keys()) & set(errors_2.keys()) }
            only_ef1_errors={k:errors_1[k] for k in set(errors_1.keys()) - set(errors_2.keys())}
            only_ef2_errors={k:errors_2[k] for k in set(errors_2.keys()) - set(errors_1.keys())}

            with open(output_file, 'w') as output:
                output.write('\nboth ({0}):\n===============\n'.format(len(errors_both)))
                for k in sorted(errors_both.keys()):
                    output.write(k + "\t" + errors_both[k] + "\n")


                output.write('\nonly in ' + error_file_1 + ' ({0}):\n===============\n'.format(len(only_ef1_errors)))
                v_lines, adj_lines, n_lines = group_by_pos(only_ef1_errors.keys())
                output.write('\nverb errors ({0})\n=======\n'.format(len(v_lines)))
                for l in sorted(v_lines):
                    output.write(l + "\t" + only_ef1_errors[l] +"\n")
                output.write('\nadj errors ({0})\n=======\n'.format(len(adj_lines)))
                for l in sorted(adj_lines):
                    output.write(l + "\t" + only_ef1_errors[l] +"\n")
                output.write('\nnoun errors ({0})\n=======\n'.format(len(n_lines)))
                for l in sorted(n_lines):
                    output.write(l + "\t" + only_ef1_errors[l] +"\n")


                output.write('\nonly in ' + error_file_2 + '({0}):\n===============\n'.format(len(only_ef2_errors)))
                v_lines, adj_lines, n_lines = group_by_pos(only_ef2_errors)
                output.write('\nverb errors ({0})\n=======\n'.format(len(v_lines)))
                for l in sorted(v_lines):
                    output.write(l + "\t" + only_ef2_errors[l] +"\n")
                output.write('\nadj errors ({0})\n=======\n'.format(len(adj_lines)))
                for l in sorted(adj_lines):
                    output.write(l + "\t" + only_ef2_errors[l] +"\n")
                output.write('\nnoun errors ({0})\n=======\n'.format(len(n_lines)))
                for l in sorted(n_lines):
                    output.write(l + "\t" + only_ef2_errors[l] +"\n")

#                output.writelines(lines)
                print 'wrote comparison to {0}'.format(output_file)
            return


def group_by_pos(lines):
    v_lines = []
    adj_lines = []
    n_lines = []
    for line in lines:
        #if 'pos=V' in line:
        if 'V;' in line:
            v_lines.append(line)
        #if 'pos=ADJ' in line:
        if 'ADJ;' in line:
            adj_lines.append(line)
        #if 'pos=N' in line:
        if 'N;' in line:
            n_lines.append(line)
    return v_lines, adj_lines, n_lines

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    # default values
    if arguments['--results_path']:
        results_dir_param = arguments['--results_path']
    else:
        results_dir_param = RESULTS_PATH
    if arguments['--data_path']:
        sigmorphon_root_dir_param = arguments['--data_path']
    else:
        sigmorphon_root_dir_param = DATA_PATH
    if arguments['--prefix']:
        prefix_param = arguments['--prefix']
    else:
        print 'prefix is mandatory'
        raise ValueError
    if arguments['--task']:
        task_param = arguments['--task']
    else:
        task_param = '1'
    if arguments['--langs']:
        langs_param = [l.strip() for l in arguments['--langs'].split(',')]
    else:
        langs_param = LANGS
    if arguments['--regimes']:
        regime_param = [l.strip() for l in arguments['--regimes'].split(',')]
    else:
        regime_param = REGIME
    if arguments['--compare']:
        compare_param = True
    else:
        compare_param = False
    if arguments['--prefix_2']:
        prefix_2_param = arguments['--prefix_2']
    else:
        if compare_param:
            print 'prefix is mandatory in compare setting'
            raise ValueError
        else:
            prefix_param_2 = None
    if arguments['--results_path_2']:
        results_dir_2_param = arguments['--results_path_2']
    else:
        results_dir_2_param = RESULTS_PATH
    if arguments['--format_2016']:
        format_2016_param = True
    else:
        format_2016_param = False

    if arguments['--format_2016_2']:
        format_2016_2_param = True
    else:
        format_2016_2_param = False


    if compare_param:
        compare_error_analysis(results_dir_param, sigmorphon_root_dir_param, langs_param, prefix_param, regime_param, task_param, results_dir_2_param, prefix_2_param, format_2016_param, format_2016_2_param)
    else:
        main(results_dir_param, sigmorphon_root_dir_param, langs_param, prefix_param, regime_param, task_param, format_2016_param)

