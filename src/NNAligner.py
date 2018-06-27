from __future__ import division
import os
import sys
import traceback
import numpy as np
import dynet as dy

from stackLSTM_parser import Vocab

class SVocab(Vocab):
    """Vocab that takes in a list of items and produces
       indexes that are less than vocab size."""
    def __init__(self, w2i):
        super(SVocab, self).__init__(w2i)

    @classmethod
    def from_list(cls, words):
        w2i = {}
        idx = 0
        for word in set(words):
            w2i[word] = idx
            idx += 1
        return SVocab(w2i)


# alignment chars
START_CHAR = 'S'
EPS_L_CHAR = '~L'
EPS_F_CHAR = '~F'
PIPE_CHAR = '|'
TILDE = '~'

START, EPS_L, EPS_F, PIPE = range(4)
ALIGN_i2w = [START_CHAR, EPS_L_CHAR, EPS_F_CHAR, PIPE_CHAR]
ALIGN_w2i = {c : i for i, c in enumerate(ALIGN_i2w)}

class NNAligner(object):
    """Encoder-decoder aligner."""
    def __init__(self, model, train_data, *args, **kwargs):
        
        # No UNK char
        vocab = SVocab.from_list(''.join(train_data.lemmas + train_data.words))
        self.w2i = vocab.w2i
        self.i2w = vocab.i2w
        self.NUM_CHARS = len(self.w2i)
        # some defaults here
        self.INPUT_DIM = kwargs.get('INPUT_DIM', 50)
        self.HIDDEN_DIM = kwargs.get('HIDDEN_DIM', 100)
        self.ACT_DIM = kwargs.get('ACT_DIM', 20)
        self.NUM_ACTS = len(ALIGN_i2w)
        
        # building a model
        self.CHAR_LOOKUP = model.add_lookup_parameters((self.NUM_CHARS, self.INPUT_DIM))
        self.ACT_LOOKUP = model.add_lookup_parameters((self.NUM_ACTS, self.ACT_DIM)) # input, output
        # lemma
        self.fLRNN = dy.LSTMBuilder(1, self.INPUT_DIM, self.HIDDEN_DIM, model)
        self.bLRNN = dy.LSTMBuilder(1, self.INPUT_DIM, self.HIDDEN_DIM, model)
        # form
        self.fFRNN = dy.LSTMBuilder(1, self.INPUT_DIM, self.HIDDEN_DIM, model)
        self.bFRNN = dy.LSTMBuilder(1, self.INPUT_DIM, self.HIDDEN_DIM, model)
        # decoder: Takes bi-enc L_i and F_j        
        self.dec = dy.LSTMBuilder(1, 4*self.HIDDEN_DIM + self.ACT_DIM, self.HIDDEN_DIM, model)
        # softmax
        self.W = model.add_parameters((self.NUM_ACTS, self.HIDDEN_DIM)) # output, input
        self.b = model.add_parameters(self.NUM_ACTS)
        
    def align(self, pair, actions=None, sampling=False, aligns_as_text=False, external_cg=False):        
        def valid_actions():
            # pretty stupid function, most of the time,
            # all actions (except Start) are valid.
            valid_actions = []
            L_diff = (L_len - L_index > 0)  # this is correct!
            F_diff = (F_len - F_index > 0)
            if L_diff: valid_actions.append(EPS_L)
            if F_diff: valid_actions.append(EPS_F)
            if L_diff and F_diff: valid_actions.append(PIPE)
            # START is never valid
            return valid_actions
        
        def encode(w, lemma=True):
            # encode word via an embedding layer and a bi-directional RNN
            w_enc = [self.CHAR_LOOKUP[self.w2i[c]] for c in w]
            if lemma:
                FRNN, BRNN = self.fLRNN, self.bLRNN
            else:
                FRNN, BRNN = self.fFRNN, self.bFRNN
            fs = FRNN.initial_state()
            bs = BRNN.initial_state()
            w_benc = [dy.concatenate(list(s)) for s in zip(fs.transduce(w_enc),
                                                           bs.transduce(reversed(w_enc)))]
            return w_benc
        
        try:
            lemma, form = pair
            
            if actions:
                actions = [a for a in actions]  # the smart way...?

            if not external_cg:
                dy.renew_cg()
            # encode pair
            L = encode(lemma, lemma=True)
            F = encode(form, lemma=False)

            # initialize indices
            L_index = 0
            F_index = 0

            L_len = len(lemma)
            F_len = len(form)

            # initialize decoder
            s = self.dec.initial_state()
            action = START

            # initialize classifier weights
            W = dy.parameter(self.W)
            b = dy.parameter(self.b)

            action_hist = []
            MAXLEN = 100
            losses = []
            # stop when both are indices reach their resp. limits
            while (L_index < L_len or F_index < F_len) and len(action_hist) < MAXLEN:
                val_actions = valid_actions()
                assert len(val_actions) > 0
                if len(val_actions) > 1:
                    # model nondeterministic choice
                    x = dy.concatenate([L[L_index], F[F_index], self.ACT_LOOKUP[action]])
                    s = s.add_input(x)
                    logits = W * s.output() + b
                    dist = dy.log_softmax(logits, val_actions)
                    if not actions:
                        np_dist = np.exp(dist.npvalue())
                        if sampling:
                            # sample according to softmax
                            rand = np.random.rand()
                            for action, p in enumerate(np_dist):
                                rand -= p
                                if rand <= 0: break
                        else:
                            # greedy
                            action = np.argmax(np_dist)
                    else:
                        # if there are oracle actions...
                        action = actions.pop(0) # pops from top
                        assert action in val_actions
                    loss = dy.pick(-dist, action)
                    losses.append(loss)
                else:
                    # otherwise noting to model
                    assert len(val_actions) == 1
                    action = val_actions[0]
                    if actions:
                        # this must also the oracle action
                        assert action == actions.pop(0)

                action_hist.append(action)
                if action == PIPE:
                    L_index += 1
                    F_index += 1
                elif action == EPS_L:
                    L_index += 1
                else:
                    F_index += 1
                    assert action == EPS_F
                    # can't be START
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print 'An error occurred on line {} in statement {}'.format(line, text)
            print 'pair: ', pair
            print 'val_actions, action, actions: ', val_actions, action, actions
            print 'indices: ', L_index, L_len, F_index, F_len
            sys.exit(1)
        # aligns_as_text=True returns an aligned pair of lemma, form, otherwise action sequence  
        return (align_pair(action_hist, lemma, form) if aligns_as_text else action_hist,
                dy.average(losses))
    
    @staticmethod
    def get_labels(aligner, pairs, **kwargs):
        aligned_pairs = aligner(pairs, **kwargs)
        labels = [generate_labels(*p) for p in aligned_pairs]
        return labels

def generate_labels(L, F, as_text=False):
    # from aligned words to labels
    labels = []
    for i, l in enumerate(L):
        f = F[i]
        if l == TILDE:
            a = EPS_F  # ~F
        elif f == TILDE:
            a = EPS_L  # ~L
        else:
            a = PIPE
        labels.append(a)
    return u''.join([ALIGN_i2w[l] for l in labels]) if as_text else labels

def align_pair(labels, L, F):
    # from labels and words, to aligned words
    L, F = list(L), list(F)
    aligned_L = []
    aligned_F = []
    for a in labels:
        if a == PIPE:
            aligned_L.append(L.pop(0))
            aligned_F.append(F.pop(0))
        elif a == EPS_F:
            # a char in the form is aligned to the empty string
            aligned_L.append(TILDE)
            aligned_F.append(F.pop(0))
        elif a == EPS_L:
            # a char in the lemma is aligned to the empty string
            aligned_L.append(L.pop(0))
            aligned_F.append(TILDE)
    return u''.join(aligned_L), u''.join(aligned_F)


if __name__ == "__main__":
    
    from transition_inflector import ActionDataSet, smart_align #, dumb_align
    from defaults import DATA_PATH
    # test whether the model performs as expected
    train_path = os.path.join(DATA_PATH, 'task1/russian-train-low')
    train_data = ActionDataSet.from_file(train_path)
    pairs = zip(train_data.lemmas, train_data.words)

    model = dy.Model()
    fancy_aligner = NNAligner(model, train_data)
    labels = fancy_aligner.get_labels(lambda x: smart_align(x, iterations=5), pairs)
    train = zip(pairs, labels)
    
    #for (L, F), actions in train[:3]:
    #    print L
    #    print F
    #    print ''.join([ALIGN_i2w[a] for a in actions])
    #    print
    
    print '\n\n### TEST 1: Train and sample from softmax'
    NUM_EPOCHS = 2
    trainer = dy.AdadeltaTrainer(model)
    train_sanity = train[:10]

    for e in range(NUM_EPOCHS):
        #np.random.shuffle(train)
        total_loss = 0.0
        for i, (pair, actions) in enumerate(train):
            action_hist, loss = fancy_aligner.align(pair, actions)
            total_loss += loss.scalar_value()  # supervised pretraining
            loss.backward()
            trainer.update()  # update after each alignment decision
        acc = 0.0
        for pair, actions in train_sanity:
            action_hist, _ = fancy_aligner.align(pair, sampling=True)
            L, F = align_pair(action_hist, pair[0], pair[1])
            print L, F
            if action_hist == actions:
                acc += 1
                print 'V'
            else: print 'X'
        print '\nEpoch %s, loss %f, sanity acc %f\n' % (e, total_loss, acc / 10.)

    # sample some alignments
    print '\n\n### TEST 2: Sample some alignments'
    import random
    [(pair, _)] = random.sample(train, 1)
    for _ in range(20):
        (F, L), _ = fancy_aligner.align(pair, sampling=True, aligns_as_text=True)
        print F, L
    
    # generate_labels and align_pair work as expected
    print '\n\n### TEST 3: Check oracle alignment action generation'
    test_pairs = [['dogged', 'dog'], ['dog', 'dogged'], ['walk', 'walking'], ['walk', 'bewalking']]
    test_aligned = smart_align(test_pairs, iterations=5)
    for (L, F), (a_L, a_F) in zip(test_pairs, test_aligned):
        print 'Gold alignment: ', a_L, a_F
        labels = generate_labels(a_L, a_F)
        print 'Oracle actions: ', labels
        aligned_L, aligned_F = align_pair(labels, L, F)
        try:
            assert aligned_L == a_L and aligned_F == a_F
            print 'Restored alignment from actions!\n'
        except AssertionError, e:
            print 'Failed to resore alignment from oracle actions!', L, F, labels
            raise e