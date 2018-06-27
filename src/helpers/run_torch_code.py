import sys
import os

def main(args):

    prefix = 'toy'

    os.system("python prepare_sigmorphon_data.py \
    /Users/roeeaharoni/research_data/sigmorphon2016-master/data/arabic-task1-train \
    /Users/roeeaharoni/research_data/sigmorphon2016-master/data/arabic-task1-dev {0} {0}_alphabet.txt True".format(prefix))

    os.chdir("/Users/roeeaharoni/GitHub/morphology-reinflection/seq2seq/")
    os.system("th run.lua \
    -wordsTrainFile /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/{0}.train.ind.word \
    -type double \
    -wordsTestFile /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/{0}.dev.ind.word \
    -lemmasTrainFile /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/{0}.train.ind.lemma \
    -lemmasTestFile /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/{0}.dev.ind.lemma \
    -featsTrainFile /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/{0}.train.ind.feats \
    -featsTestFile /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/{0}.dev.ind.feats \
    -alphabet /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/{0}_alphabet.txt".format(prefix))

    #os.system("python prepare_sigmorphon_eval.py <sigmorphon gold file> <torch pred file> <sigmorphon pred file>")

if __name__ == '__main__':
    main(sys.argv)



