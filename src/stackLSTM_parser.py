from __future__ import division
from operator import itemgetter
from itertools import count
from collections import Counter, defaultdict
import random
import numpy as np
import dynet as dy
import re

WORD_DIM = 64
LSTM_DIM = 64
ACTION_DIM = 32


# represents a bidirectional mapping from strings to ints
class Vocab(object):
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}

    @classmethod
    def from_list(cls, words):
        w2i = {}
        idx = 0
        for word in words:
            w2i[word] = idx
            idx += 1
        return Vocab(w2i)

    @classmethod
    def from_file(cls, vocab_fname):
        words = []
        with file(vocab_fname) as fh:
            for line in fh:
                line.strip()
                word, count = line.split()
                words.append(word)
        return Vocab.from_list(words)

    def size(self): return len(self.w2i.keys())


# In[5]:

# format:
# John left . ||| SHIFT SHIFT REDUCE_L SHIFT REDUCE_R
def read_oracle(fname, vw, va):
    with file(fname) as fh:
        for line in fh:
            line = line.strip()
            ssent, sacts = re.split(r' \|\|\| ', line)
            sent = [vw.w2i[x] for x in ssent.split()]
            acts = [va.w2i[x] for x in sacts.split()]
            sent.reverse()
            acts.reverse()
            yield (sent, acts)


# In[6]:

class StackRNN(object):
    def __init__(self, rnn, p_empty_embedding = None):
        self.s = [(rnn.initial_state(), None)]
        self.empty = None
        if p_empty_embedding:
            self.empty = dy.parameter(p_empty_embedding)
    def push(self, expr, extra=None):
        self.s.append((self.s[-1][0].add_input(expr), extra))
    def pop(self):
        return self.s.pop()[1] # return "extra" (i.e., whatever the caller wants or None)
    def embedding(self):
        # work around since inital_state.output() is None
        return self.s[-1][0].output() if len(self.s) > 1 else self.empty
    def __len__(self):
        return len(self.s) - 1


# In[7]:

# actions the parser can take
acts = ['SHIFT', 'REDUCE_L', 'REDUCE_R']
vocab_acts = Vocab.from_list(acts)
SHIFT = vocab_acts.w2i['SHIFT']
REDUCE_L = vocab_acts.w2i['REDUCE_L']
REDUCE_R = vocab_acts.w2i['REDUCE_R']
NUM_ACTIONS = vocab_acts.size()

class TransitionParser(object):
    def __init__(self, model, vocab):
        self.vocab = vocab
        # syntactic composition
        self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
        self.pb_comp = model.add_parameters(LSTM_DIM)
        # parser state to hidden
        self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
        self.pb_s2h = model.add_parameters(LSTM_DIM)
        # hidden to action
        self.pW_act = model.add_parameters((NUM_ACTIONS, LSTM_DIM))
        self.pb_act = model.add_parameters(NUM_ACTIONS)

        # layers, in-dim, out-dim, model
        self.buffRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
        self.stackRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
        self.pempty_buffer_emb = model.add_parameters(LSTM_DIM)
        self.WORDS_LOOKUP = model.add_lookup_parameters((vocab.size(), WORD_DIM))

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def parse(self, tokens, oracle_actions=None):
        def _valid_actions(stack, buffer):
            valid_actions = []
            if len(buffer) > 0:
                valid_actions += [SHIFT]
            if len(stack) >= 2:
                valid_actions += [REDUCE_L, REDUCE_R]
            return valid_actions

        dy.renew_cg() # each sentence gets its own graph
        if oracle_actions: oracle_actions = list(oracle_actions)
        buffer = StackRNN(self.buffRNN, self.pempty_buffer_emb)
        stack = StackRNN(self.stackRNN)
    
        # Put the parameters in the cg
        W_comp = dy.parameter(self.pW_comp) # syntactic composition
        b_comp = dy.parameter(self.pb_comp)
        W_s2h = dy.parameter(self.pW_s2h)   # state to hidden
        b_s2h = dy.parameter(self.pb_s2h)
        W_act = dy.parameter(self.pW_act)   # hidden to action
        b_act = dy.parameter(self.pb_act)
    
        # We will keep track of all the losses we accumulate during parsing.
        # If some decision is unambiguous because it's the only thing valid given
        # the parser state, we will not model it. We only model what is ambiguous.
        losses = []
        
        # push the tokens onto the buffer (tokens is in reverse order)
        for tok in tokens:
            tok_embedding = self.WORDS_LOOKUP[tok]
            buffer.push(tok_embedding, (tok_embedding, self.vocab.i2w[tok]))

        while not (len(stack) == 1 and len(buffer) == 0):
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            valid_actions = _valid_actions(stack, buffer)
            log_probs = None
            action = valid_actions[0]
            if len(valid_actions) > 1:
                p_t = dy.concatenate([buffer.embedding(), stack.embedding()])
                h = dy.tanh(W_s2h * p_t + b_s2h)
                logits = W_act * h + b_act
                log_probs = dy.log_softmax(logits, valid_actions)
                if oracle_actions is None:
                    action = np.argmax(log_probs.npvalue())
            if oracle_actions is not None:
                action = oracle_actions.pop()
            if log_probs is not None:
                # append the action-specific loss
                losses.append(dy.pick(log_probs, action))

            # execute the action to update the parser state
            if action == SHIFT:
                tok_embedding, token = buffer.pop()
                stack.push(tok_embedding, (tok_embedding, token))
            else: # one of the REDUCE actions
                right = stack.pop() # pop a stack state
                left = stack.pop()  # pop another stack state
                # figure out which is the head and which is the modifier
                head, modifier = (left, right) if action == REDUCE_R else (right, left)
        
                # compute composed representation
                head_rep, head_tok = head
                mod_rep, mod_tok = modifier
                composed_rep = dy.tanh(W_comp * dy.concatenate([head_rep, mod_rep]) + b_comp)
                
                stack.push(composed_rep, (composed_rep, head_tok))
                if oracle_actions is None:
                    print('{0} --> {1}'.format(head_tok, mod_tok))

        # the head of the tree that remains at the top of the stack is the root
        if oracle_actions is None:
            head = stack.pop()[1]
            print('ROOT --> {0}'.format(head))
        return -dy.esum(losses) if losses else None


# In[8]:

if __name__ == "__main__":

    # load training and dev data
    vocab_words = Vocab.from_file('data/vocab.txt')
    train = list(read_oracle('data/small-train.unk.txt', vocab_words, vocab_acts))
    dev = list(read_oracle('data/small-dev.unk.txt', vocab_words, vocab_acts))

    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    tp = TransitionParser(model, vocab_words)


    # In[9]:

    instances_processed = 0
    validation_losses = []
    for epoch in range(5):
        random.shuffle(train)
        words = 0
        total_loss = 0.0
        for (s,a) in train:
            # periodically report validation loss
            e = instances_processed / len(train)
            if instances_processed % 1000 == 0:
                dev_words = 0
                dev_loss = 0.0
                for (ds, da) in dev:
                    loss = tp.parse(ds, da)
                    dev_words += len(ds)
                    if loss is not None:
                        dev_loss += loss.scalar_value()
                print('[validation] epoch {}: per-word loss: {}'.format(e, dev_loss / dev_words))
                validation_losses.append(dev_loss)

            # report training loss
            if instances_processed % 100 == 0 and words > 0:
                print('epoch {}: per-word loss: {}'.format(e, total_loss / words))
                words = 0
                total_loss = 0.0

            # here we do training
            loss = tp.parse(s, a) # returns None for 1-word sentencs (it's clear how to parse them)
            words += len(s)
            instances_processed += 1
            if loss is not None:
                total_loss += loss.scalar_value()
                loss.backward()
                trainer.update()


    # In[10]:

    s = 'Parsing in Austin is fun .'
    UNK = vocab_words.w2i['<unk>']
    toks = [vocab_words.w2i[x] if x in vocab_words.w2i else UNK for x in s.split()]
    toks.reverse()
    tp.parse(toks)