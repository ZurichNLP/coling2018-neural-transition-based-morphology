# show V's and X's on predicted output file according to gold file
import math
def main():

    langs = ['arabic', 'finnish', 'georgian', 'russian', 'german', 'turkish', 'spanish', 'navajo', 'hungarian', 'maltese']
    tasks = ['1', '2', '3']
    models = ['blstm', 'nfst']

    solution_input_file_format = '/Users/roeeaharoni/GitHub/sigmorphon2016/data/{0}-task{1}-test-covered'

    solution_output_file_format = \
        '/Users/roeeaharoni/GitHub/morphological-reinflection/results/solutions/{0}/{1}-task{2}-solution'

    nbest_solution_output_file_format = \
        '/Users/roeeaharoni/GitHub/morphological-reinflection/results/solutions/{0}/nbest/{1}-task{2}-solution'


    # check for lines amount
    i=0
    for lang in langs:
        for task in tasks:
            input_file_path = solution_input_file_format.format(lang, task)
            for model in models:
                if task=='3' and model=='nfst':
                    continue
                output_file_path = solution_output_file_format.format(model,lang,task)
                with open(input_file_path) as input:
                    input_lines = input.readlines()

                    with open(output_file_path) as output:
                        output_lines = output.readlines()
                        for l, line in enumerate(output_lines):
                            if ((task == '1' or task =='3') and len(line.split()) != 3) or (task=='2' and len(line.split()) != 4) :
                                print 'bad line in file {0} line num {1}'.format(output_file_path, l)
                                print 'split len is {0}\n'.format(len(line.split()))
                        if len(output_lines) != len(input_lines):
                            print 'mismatch in {0} {1} {2} vs. {3}'.format(input_file_path,
                                                                           output_file_path,
                                                                           len(input_lines),
                                                                           len(output_lines))
                        else:
                            print '{0} {1} {2} OK'.format(lang, task, model)
                            i += 1
    print '{0} files passed'.format(i)

    print 'now checking nbest files'
    # check for lines amount in nbest
    i = 0
    tasks=['1','2','3']
    models= ['blstm']

    for lang in langs:
        for task in tasks:
            input_file_path = solution_input_file_format.format(lang, task)
            for model in models:
                if model == 'nfst':
                    continue
                nbest_output_file_path = nbest_solution_output_file_format.format(model, lang, task)
                output_file_path = solution_output_file_format.format(model,lang,task)

                with open(input_file_path) as input:
                    input_lines = input.readlines()

                    with open(output_file_path) as output:
                        output_lines = output.readlines()
                    with open(nbest_output_file_path) as nbest_output:
                        nbest_output_lines = nbest_output.readlines()
                        for l, line in enumerate(nbest_output_lines):

                            if l%5 == 0:
                                # check that first hypo is not empty
                                if task !='2' and len(nbest_output_lines[l].split()) != 3:
                                    print 'EMPTY FIRST HYPO AT {0} LINE {1}'.format(nbest_output_file_path, l)

                                # check consistency between greedy and nbest
                                if nbest_output_lines[l] != output_lines[l/5]:
                                    print 'inconsistency between nbest and greedy:\n {0} line {1} \n {2} line {3}\n nbest: {4} \n greedy: {5}'.format(
                                        nbest_output_file_path,
                                        l,
                                        output_file_path,
                                        l/5,
                                        nbest_output_lines[l], output_lines[l/5])


                            # check if no empty predictions
                            if ((task == '1' or task == '3') and len(line.split()) != 3) or (
                                    task == '2' and len(line.split()) != 4):

                                print 'bad line in file {0} line num {1}'.format(output_file_path, l)
                                print 'split len is {0}\n'.format(len(line.split()))

                        # check if line amount is correct
                        if len(nbest_output_lines) != 5*len(input_lines):
                            print 'mismatch in {0} {1} {2} vs. {3}'.format(input_file_path,
                                                                           output_file_path,
                                                                           len(input_lines),
                                                                           len(nbest_output_lines))
                        else:
                            print '{0} {1} {2} OK'.format(lang, task, model)
                            i += 1
    print '{0} files passed'.format(i)


    return


if __name__ == '__main__':
    main()