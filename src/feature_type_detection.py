from __future__ import division
import os
import glob
import re
import sys
import csv
from collections import defaultdict, Counter

from defaults import DATA_PATH

def find_tagseqs(fns, filter_cond=lambda x: x.endswith('dev')):
    """Returns a dictionary from language to lists of
    tag sequences appearing in files possibly filtered
    with some filtering condition, e.g. that this should
    not be a dev set file."""
    all_tags = defaultdict(list)
    for fn in fns:
        tmp = []
        if filter_cond(fn):
            continue
        with open(fn) as f:
            for lemma, form, features in csv.reader(f, delimiter='\t'):
                tmp.append(features)
        lang = os.path.basename(fn).rsplit('-', 2)[0]
        all_tags[lang].extend(tmp)
    return all_tags

#### SOME FEATURE TYPES FROM John Sylak-Glassman, 2016 WITH CORRECTIONS BASED ON DATA ####

QUECHUAN_CASE = ['DISTR', 'EXCLV', 'CAUSV']
QUECHUAN_ADJ = ['BEADJ', 'GEADJ']
ARABIC_NDEF = ['NDEF']  # looks like = INDF
TURKISH_NOMINAL_PERS_NUMBER = ['1S', '2S', '3S', '1P', '2P', '3P'] # occurs on nouns,
                                                                   # together with normal NUMBER marking
TURKISH_LOC = ['LOC']
# Armenian: V.CONV converb, RES resultative participle, SUB subject participle, FUT1, FUT2 converbs, CAUSE causative
INST = ['INST']  # = INS
NAVAJO_POSS = ['PSS4', 'PSS3', 'PSS0']
ARMENIAN_CONVERB = ['V.CONV']
#ARMENIAN_NONFIN = ['SUB', 'RES', 'CAUSE'] CONN...?

FEATS = {'Aspect' : ['HAB', 'IPFV', 'ITER', 'PFV', 'PRF', 'PROG', 'PROSP'],
         'Case'   : ['ABL', 'ABS', 'ACC', 'ALL', 'ANTE', 'APPRX', 'APUD',
                     'AT', 'AVR', 'BEN', 'BYWAY', 'CIRC', 'COM', 'COMPV',
                     'DAT', 'EQTV', 'ERG', 'ESS', 'FRML', 'GEN', 'IN', 'INS',
                     'INTER', 'NOM', 'NOMS', 'ON', 'ONHR', 'ONVR', 'POST',
                     'PRIV', 'PROL', 'PROPR', 'PROX', 'PRP', 'PRT', 'REL',
                     'REM', 'TERM', 'TRANS', 'VERS', 'VOC'] + QUECHUAN_CASE + TURKISH_LOC + INST,
         'Argument Marking' : ['ARGAC3S'],
         'Animacy' : ['ANIM', 'INAN', 'HUM', 'NHUM'],
         'Definiteness' : ['DEF', 'INDF', 'SPEC', 'NSPEC'] + ARABIC_NDEF,
         'Tense'   : ['1DAY', 'FUT', 'HOD', 'IMMED', 'PRS', 'PST', 'RCT', 'RMT'],
         'Mood'    : ['ADM', 'AUNPRP', 'AUPRP', 'COND', 'DEB', 'DED', 'IMP',
                      'IND', 'INTEN', 'IRR', 'LKLY', 'OBLIG', 'OPT', 'PERM',
                      'POT', 'PURP', 'REAL', 'SBJV', 'SIM'],
         'Gender'  : ['BANTU1-23', 'FEM', 'MASC', 'NAKH1-8', 'NEUT'],
         'Number'  : ['DU', 'GPAUC', 'GRPL', 'INVN', 'PAUC', 'PL', 'SG'],
         'Comparision' : ['AB', 'CMPR', 'EQT', 'RL', 'SPRL'],
         'Deixis'  : ['ABV', 'BEL', 'EVEN', 'MED', 'NOREF', 'NVIS',
                      'PHOR', 'PROX', 'REF1', 'REF2', 'REMT', 'VIS'],
         'Evidentiality' : ['ASSUM', 'AUD', 'DRCT', 'FH', 'HRSY', 'INFER',
                            'NFH', 'NVSEN', 'QUOT', 'RPRT', 'SEN'],
         'Finiteness' : ['FIN', 'NFIN'],
         'Information Structure' : ['FOC', 'TOP'],
         'Interrogativity' : ['DECL', 'INT'],
         'POS'  : ['ADJ', 'ADP', 'ADV', 'ART', 'AUX', 'CLF',   # Part of Speech
                   'COMP', 'CONJ', 'DET', 'INTJ', 'N', 'NUM',
                   'PART', 'PRO', 'PROPN', 'V', 'V.CVB',
                   'V.MSDR', 'V.PTCP'] + ARMENIAN_CONVERB,
         'Person'  : ['0', '1', '2', '3', '4', 'EXCL', 'INCL', 'OBV', 'PRX'],
         'Polarity' : ['POS', 'NEG'],
         'Politeness' : ['AVOID', 'COL', 'ELEV', 'FOREG', 'FORM', 'HIGH',
                         'HUMB', 'INFM', 'LIT', 'LOW', 'POL', 'STELEV',
                         'STSUPR'],
         'Possession' : ['ALN', 'NALN', 'PSS1D', 'PSS1DE', 'PSS1DI', 'PSS1P',
                         'PSS1PE', 'PSS1PI', 'PSS1S', 'PSS2D', 'PSS2DF',
                         'PSS2DM', 'PSS2P', 'PSS2PF', 'PSS2PM', 'PSS2S',
                         'PSS2SF', 'PSS2SFORM', 'PSS2SINFM', 'PSS2SM',
                         'PSS3D', 'PSS3DF', 'PSS3DM', 'PSS3P', 'PSS3PF',
                         'PSS3PM', 'PSS3S', 'PSS3SF', 'PSS3SM', 'PSSD'] + NAVAJO_POSS,
         'Switch-Reference' : ['CN', 'R', 'MN', 'DS', 'DSADV', 'LOG',
                               'OR', 'SEQMA', 'SIMMA', 'SS', 'SSADV'],
         'Valency'    : ['APPL', 'CAUS', 'DITR', 'IMPRS', 'INTR', 'RECP',
                         'REFL', 'TR'],
         'Voice'      : ['ACFOC', 'ACT', 'AGFOC', 'ANTIP', 'BFOC', 'CFOC',
                         'DIR', 'IFOC', 'INV', 'LFOC', 'MID', 'PASS', 'PFOC'],
         'Quechuan Adj Features' : QUECHUAN_ADJ,
         'Turkish Mominal PersNum' : TURKISH_NOMINAL_PERS_NUMBER,
        }

REVERSE_FEATS = {v : k for k, vs in FEATS.items() for v in vs}


def make_feat_dict(tag_seq_, lang,
                   return_funny_features=False,
                   verbose=False):

    funny_features = []  # unexpected features
    
    # 1. features are conjoined with ";".
    # One occasionally sees ":", e.g. in Persian: "2:PL" for "2;PL" or "PRF:COL" for "PRF;COL".
    # On the other hand, "1;PL" and "IPFV;COL" appear in Persian data.
    tag_seq = re.split('[;:]', tag_seq_)
    
    # 2. POS tag is normally the first feature. In some languages, V is the first POS tag,
    # and then comes patriciple, etc. tag, which should actually be the POS tag.
    pos_tag = tag_seq[0]
    word_feats = {'POS' : pos_tag}
    
    # 3. some feature values appear multiple times! (e.g. Albanian PRF;PRF)
    for t in set(tag_seq[1:]):
        # 4. look up feature type, if applies
        feature_type = REVERSE_FEATS.get(t)
        if feature_type:
            # are we not overwriting any feature value?
            if feature_type in word_feats:
                if lang == 'hebrew' and feature_type == 'Possession' and t == 'PSSD':
                    # 5. Hebrew marks possession with 'PSSD' and again more specifically
                    # with smth like 'PSS1S'. Ignore the most generic possession marker.
                    continue
                elif lang == 'macedonian' and feature_type == 'Aspect':
                    # 6. Macedonian marks 2 aspects, one of which is PROG
                    #print word_feats[feature_type], t
                    assert word_feats[feature_type] == 'PROG'
                    word_feats['PROG'] = 'T'
                    word_feats['Aspect'] = t
                else:
                    print ('Feature conflict! %s: New value %s, current value %s '
                           'for feature type %s in feature sequence %s.' %
                           (lang, t, word_feats[feature_type], feature_type, tag_seq_))
            word_feats[feature_type] = t
        elif t.upper().startswith('LGSPEC'):
            # 7. special language features, indicate that binary feature is on
            word_feats[t.upper()] = 'T'
        elif '+' in t or '/' in t:
            # 8. conjoined and disjoint features,
            # try to find feature type using any of the features
            tt = re.split('[+/]', t)
            #assert len(tt) == 2
            feature_type = [ftype for t_ in tt for ftype in (REVERSE_FEATS.get(t_),) if ftype][0]
            word_feats[feature_type] = t
        elif lang == 'hebrew' and t.endswith('DEF'):
            # 9. Hebrew has PLDEF and SGDEF which should simply be DEF, because
            # PL and SG are already among features
            word_feats['Definiteness'] = 'DEF'
        else:
            funny_features.append(t)
            word_feats[t] = 'T'
    #word_feats = sorted(word_feats.items())
    if return_funny_features:
        if verbose:
            print ('%s: Some unexpected features in feature sequence %s: %s'
                   % (lang, tag_seq_, funny_features))
        out = word_feats, funny_features
    else:
        out = word_feats
    return out


def feature_stats(all_tags,
                  return_funny_features=True):

    all_feature_seqs = {}
    funny_features = defaultdict(set)
    for lang, ltags in all_tags.iteritems():
        tmp = []
        for tag_seq in ltags:
            word_feats, ff = make_feat_dict(tag_seq, lang, 
                                           return_funny_features)
            tmp.append(word_feats)
            funny_features[lang].update(ff)
        all_feature_seqs[lang] = tmp
    
    stats = []
    for lang, fseqs in all_feature_seqs.items():
        # shows only `langs` languages
        # if langs != 'all' and lang not in langs:
        #     continue
        l_fseqs = len(fseqs)
        rhalf_l_fseqs = l_fseqs // 2
        sorted_ls = sorted(len(f) for f in fseqs)
        if l_fseqs % 2 == 1:
            median_fseq_size = sorted_ls[rhalf_l_fseqs + 1]
        else:
            median_fseq_size = sum(sorted_ls[rhalf_l_fseqs-1 : rhalf_l_fseqs+1]) / 2
        #c = Counter(len(f) for f in fseqs)
        #mode_fseq_size, _ = c.most_common(1)[0]
        #avg_fseq_size = sum(len(f) for f in fseqs) / l_fseqs
        fseq_types = set(tuple(sorted(f.items())) for f in fseqs)
        l_fseq_types = len(fseq_types)
        stats.append((lang, l_fseqs, l_fseq_types, median_fseq_size))

    print '\n{:20}\t{:>10}\t{:>15}\t{:>25}'.format(
        'LANGUAGE', 'SAMPLES', 'UNIQ FEAT SETS', 'MEDIAN FEAT SET SIZE')
    for t in sorted(stats, key=lambda x: x[2], reverse=True):
        print '{:20}\t{:>10}\t{:>15}\t{:>25.2f}'.format(*t)
        
    if return_funny_features:
        print
        print dict(funny_features)
    
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        langs = sys.argv[1].split(',')
        fns = [fn for l in langs
               for fn in glob.glob(os.path.join(DATA_PATH, 'task1/%s*') % l)]
    else:
        langs = 'all'
        fns = glob.glob(os.path.join(DATA_PATH, 'task1/*'))
    
    all_tags = find_tagseqs(fns)
    feature_stats(all_tags)