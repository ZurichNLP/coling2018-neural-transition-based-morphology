# mode, flags, mode name's extention, mle patience, patience

# This set-up uses optimal oracle, neg log-likelihood loss, and no model roll-outs, i.e. very much like dynamic oracles for parsing
dynamic_oracles = lambda p_mle, p: ('ss', '--ss-optimal-oracle --ss-loss=nll --ss-beta=1 --ss-global-rollout', 'd', p_mle, p)

# This set-up uses optimal oracle, neg log-likelihood loss, and model roll-outs, i.e. like LOLS
lols = lambda p_mle, p: ('ss', '--ss-optimal-oracle --ss-loss=nll --ss-beta=0.5 --ss-global-rollout', 'ro', p_mle, p)  # ro = rollouts, optimal

# This set-up is LOLS with softmax-margin:
lols_margin = lambda p_mle, p: ('ss', '--ss-optimal-oracle --ss-loss=softmax-margin --ss-beta=0.5 --ss-global-rollout', 'rom', p_mle, p)  # rom = rollouts, optimal, margin

# Add 'r' for suboptimal with isert bias

# This set-up is LOLS with softmax-margin and suboptimal oracle:
lols_sub_margin = lambda p_mle, p: ('ss', '--ss-loss=softmax-margin --ss-beta=0.5 --ss-global-rollout', 'rsm', p_mle, p)  # rsm = rollouts, suboptimal, margin

# This set-up is LOLS with softmax-margin, optimal oracle, and a bias for inserts:
lols_ins_margin = lambda p_mle, p: ('ss', '--ss-optimal-oracle --ss-loss=softmax-margin --ss-beta=0.5 --ss-global-rollout --ss-bias-inserts', 'romn', p_mle, p)  # romn = rollouts, optimal, margin, iNserts
# Add 'romr' for late roll-ins
