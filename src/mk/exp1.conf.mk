#!/usr/bin/make -rRf
EXP_MAKEFILE:=$(lastword $(MAKEFILE_LIST))

# default to low if not set from environment
# SETTING=medium make -f exp0.conf.mk cv-x-target
SETTING?=low

# output additional information
# if set to 1
CVDBG?=

# ignore the output - CHECK THE CORRECT ENVIRONMENT!
#SETPYTHONENVIRONMENT?=source anaconda2.sh > /dev/null
#SETPYTHONENVIRONMENT?=source activate py27 > /dev/null
SETPYTHONENVIRONMENT?=source ~/.virtualenvs/dynet-env/bin/activate > /dev/null

XVN2OPTIONS:= mk/exp1.xvn2options.inc.perl
RESULTSDIR?=../results

DATADIR?=../data/all/task1

EXP ?= exp1 # sollte EXP.conf.mk und matchen

# ensure that the build rules are rebuilt unconditionally
#$(warning $(realpath $(EXP_MAKEFILE)))
#$(shell $(MAKE) -f $(realpath $(EXP_MAKEFILE)) test-rules)

CONLLTASK1DATA:=conll2017/all/task1
PYTHONCALL?=python
MKDIR?=mkdir -p $(@D) 

LNG+=albanian
LNG+=armenian
LNG+=basque
LNG+=bengali
LNG+=bulgarian
LNG+=catalan
LNG+=czech
LNG+=danish
LNG+=dutch
LNG+=english
LNG+=estonian
LNG+=faroese
LNG+=finnish
LNG+=french
LNG+=georgian
LNG+=german
LNG+=haida
LNG+=hebrew
LNG+=hindi
LNG+=hungarian
LNG+=icelandic
LNG+=irish
LNG+=italian
LNG+=khaling
LNG+=kurmanji
LNG+=latin
LNG+=latvian
LNG+=lithuanian
LNG+=lower-sorbian
LNG+=macedonian
LNG+=navajo
LNG+=northern-sami
LNG+=norwegian-bokmal
LNG+=norwegian-nynorsk
LNG+=persian
LNG+=polish
LNG+=portuguese
LNG+=quechua
LNG+=romanian
LNG+=russian
ifneq ($(SETTING),high)
LNG+=scottish-gaelic
endif
LNG+=serbo-croatian
LNG+=slovak
LNG+=slovene
LNG+=sorani
LNG+=spanish
LNG+=swedish
LNG+=turkish
LNG+=ukrainian
LNG+=urdu
LNG+=welsh

# for testing
#LNG:=german
#LNG:=russian
#LNG:=turkish
#LNG:=albanian

SEED:=1

### IMPORTANT every XVNSET variable starts with a string that matches -\w !!


#XV1SET := -hacmsmrt
#XV1SET += -hacmcls
XV1SET := -haemsmrt
XV1SET += -haemcls

# Random {SEED}
#XV2SET += -s1

# {EPOCHS}
# Values for XV4
XV2SET += -e50

# {PATIENCE}
XV3SET += -p10
# by default 30

# {DROPOUT} 
XV4SET += -d0
# by default 0.5

# {BEAM}
XV5SET += -b1
#XV5SET += -b3



### test with touch mk/*meta & SETTING=??? make -f exp0.conf.mk cv-x-target -n
# up to XV9SET


include cv-make/cv-setup.mk
ifdef $(CVDBG)
$(warning XVSET, $(XVSET))
endif

###############################################################################
# START OF cv-x-target
###############################################################################

### File-List: cv-x-files
# All files for cv experimentation

# 
# x-{SYSTEM}-{EPOCHS}-{PATIENCE}-{DROPOUT}-{BEAM}-x/seed-{SEED}/{LNG}-{SETTING}.dev.eval

cv-x-files += $(foreach q,$(SETTING),$(foreach l,$(LNG),$(foreach x,$(XVSET),$(foreach s,$(SEED),$(RESULTSDIR)/$(x)/seed-$(s)/$(l)-$(q).dev.eval)))) # see exp0.rules.mk.meta
ifdef $(CVDBG)
$(warning cv-x-files:$(cv-x-files))
endif
cv-x-target: $(cv-x-files)

###############################################################################
# Create overall stats
###############################################################################

make-dev-stats:
	cd $(RESULTSDIR); \
	mkdir -p stats-$(SETTING).d; \
	for d in $$(ls -d  x* ) ; do \
		mkdir -p stats-$(SETTING).d/$$d; \
	done  ; \
	for d in $$(ls -d  x*/s* ) ; do \
		printf "LANG\t%s\n" $$d > stats-$(SETTING).d/$$d.tsv ; \
		for l in $(LNG) ; do \
			#echo $$d/$$l-$(SETTING)/*.stats ; \
			if test -e  $$d/$$l-$(SETTING)/*.stats ; then \
				printf "%s\t%1.3f\n" $$l $$(awk '/ACCURACY/ {print $$6}'  < $$d/$$l-$(SETTING)/*.stats  ); \
			else \
				printf "%s\tn/a\n" $$l ; \
			fi ; \
		done >> stats-$(SETTING).d/$$d.tsv; \
	done  ; \
	paste stats-$(SETTING).d/x-*/seed-*.tsv | perl -ln -e 's/\bx-//g;s/-x\b//g;s/LANG\t//g;s/(\t|^)[[a-z]+-?[a-z]+\t/\t/g;s/\t+/\t/g;s/^\t//; print;' | \
	python -c 'import pandas,sys;df=pandas.read_table(sys.argv[1],na_values="n/a");df.dropna(axis=1,how="all",inplace=True);df.to_csv(sys.argv[2],sep="\t",na_rep="n/a",index=False) ' /dev/stdin  stats-$(SETTING).d/all.tsv ;\
	echo "NOTE: Find all currently available results at $$(readlink -f stats-$(SETTING).d/all.tsv)"


ifdef $(CVDBG)
$(warning $(MAIN_D)/)
endif

# Make sure the directory exists
$(shell mkdir -p $(EXP_D)/mk.d/)
include $(EXP_D)/mk.d/$(EXP).Rules.mk

define x2option
$(shell perl   cv-make/lib/xvn2options.perl -file $(XVN2OPTIONS)  -- $1  2> /dev/null)
endef

# first => rst: strip first two characters (typically -a options)
define rst
$(call substr,$1,$(call strlen,$1))
endef

# Call one shell per recipe line
#.ONESHELL:


SHELL:=/bin/bash
