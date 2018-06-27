# coding: utf-8

ALIGN_SYMBOL = '~'

def main():
    lemma = u'<yūnāniyyun>'
    input =  u'~~~yūnāniyyun>'
    output = u'al-yūnāniyyi~>'

    # lemma = u'<kalb>'
    # input =  u'kalb~~>'
    # output = u'kalber>'

    # lemma = u'<ʿumāniyyun>'
    # input =  u'~~~ʿumāniyyun>'
    # output = u'al-ʿumān~~~ī~>'

    # lemma = '<lemma>'
    # input =  '~~l~emma~>'
    # output = 'dillemmos>'

    print list(input)
    print len(input)
    print list(output)
    print len(output)

    print output_nfst_steps(input, output, lemma)

def output_nfst_steps(aligned_lemma, aligned_word, lemma):
    i = 0
    j = 0
    possible_outputs = []
    for index, e in enumerate(aligned_word):

        # end of sequence - force output of end word and finish
        if e == '>':
            possible_outputs.append('>')
            print u'({0},{1}) {2}'.format(i, j, '>')
            break

        # beginning of sequence - step if there's no prefix to write
        if lemma[i] == '<' and aligned_lemma[index] != ALIGN_SYMBOL:
            # perform rnn step
            # feedback, i, j, blstm[i], feats
            possible_outputs.append('^')
            print u'({0},{1})'.format(i, j)
            i += 1

        # middle of sequence - check if there is output character to fire
        if aligned_word[index] != ALIGN_SYMBOL:

            if lemma[i] == aligned_word[index]:
                possible_outputs.append(str(i))  # copy i action - maybe model as a single action?
                #possible_outputs.append(lemma[i])
            else:
                possible_outputs.append(aligned_word[index])
            print u'({0},{1}) {2}'.format(i, j, possible_outputs[-1])
            j += 1


        # now check if it's time to progress on input (otherwise wait here)
        if i < len(lemma) - 1 and aligned_lemma[index + 1] != '~':
            # perform rnn step
            # feedback, i, j, blstm[i], feats
            possible_outputs.append('^')
            print u'({0},{1})'.format(i, j)
            i += 1
            
    return possible_outputs

if __name__ == '__main__':
    main()