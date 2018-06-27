
SUBMISSIONDIR?=../submission
GOLDDIR?=conll2017/all/task1


INSTITUTION?=CLUZH

LOW_MEMBERS:=1 2 3 4 5
MEDIUM_MEMBERS:= 1 2 3 4 5
HIGH_MEMBERS:= 1 2 3
HIGH_TRANS_DMB_MEMBERS:= 1 2

NBEST?=0
NBEST_IND?=7

# Do not waste cycles while testing the submission
GENERATE_ZIP?=0

# https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-01-max-hard
MAX_HARD:= 01

# Ensemble Members of MAX_HARD
low-$(MAX_HARD)-member    += x-shdmix-a-o-e50-h200-b1-r0-x
low-$(MAX_HARD)-member    += x-shdmixdmb-a-o-e50-h200-b1-r0-x
medium-$(MAX_HARD)-member += x-shdmix-a-o-e30-h200-b1-r0-x
medium-$(MAX_HARD)-member += x-shdmixdmb-a-o-e30-h200-b1-r0-x
high-$(MAX_HARD)-member   += x-shdmix-a-o-e20-h200-b1-r0-x
high-$(MAX_HARD)-member   += x-shdmixdmb-a-o-e20-h200-b1-r0-x

# Targets of MAX_HARD
create-submission-target += create-submission-$(MAX_HARD)-low 
create-submission-target += create-submission-$(MAX_HARD)-medium 
create-submission-target += create-submission-$(MAX_HARD)-high

# https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-01-max-hard
MAX_NEMATUS:= 10

# Ensemble Members of MAX_HARD
high-$(MAX_NEMATUS)-member   += x-snematus-a-o-e100-h600-b1-r1-x


# Targets of MAX_HARD
create-submission-target += create-submission-$(MAX_NEMATUS)-high

#  create-submission-$(MAX_NEMATUS)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_NEMATUS)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_NEMATUS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_NEMATUS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_NEMATUS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_NEMATUS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_NEMATUS)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_NEMATUS)-0/task1/00ALL-$*-dev.result.txt ;

###################################################################
# BEGIN MAX_HARDSMART
####################################################################
# Evaluate the effect of smart vs naive alignment
MAX_HARDSMART:= HS

# Ensemble Members of MAX_HARDSMART
low-$(MAX_HARDSMART)-member    += x-shdmix-a-o-e50-h200-b1-r0-x
medium-$(MAX_HARDSMART)-member += x-shdmix-a-o-e30-h200-b1-r0-x
high-$(MAX_HARDSMART)-member   += x-shdmix-a-o-e20-h200-b1-r0-x


# Targets of MAX_HARDSMART
create-submission-target += create-submission-$(MAX_HARDSMART)-low
create-submission-target += create-submission-$(MAX_HARDSMART)-medium 
create-submission-target += create-submission-$(MAX_HARDSMART)-high

#  create-submission-$(MAX_HARDSMART)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_HARDSMART)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_HARDSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_HARDSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_HARDSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_HARDSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_HARDSMART)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDSMART)-0/task1/00ALL-$*-dev.result.txt ;

###################################################################
# END MAX_HARDSMART
####################################################################

###################################################################
# BEGIN MAX_HARDNAIVE
####################################################################
# Evaluate the effect of smart vs naive alignment
MAX_HARDNAIVE:= HN

# Ensemble Members of MAX_HARDNAIVE
low-$(MAX_HARDNAIVE)-member    += x-shdmixdmb-a-o-e50-h200-b1-r0-x
medium-$(MAX_HARDNAIVE)-member += x-shdmixdmb-a-o-e30-h200-b1-r0-x
high-$(MAX_HARDNAIVE)-member   += x-shdmixdmb-a-o-e20-h200-b1-r0-x


# Targets of MAX_HARDNAIVE
create-submission-target += create-submission-$(MAX_HARDNAIVE)-low
create-submission-target += create-submission-$(MAX_HARDNAIVE)-medium 
create-submission-target += create-submission-$(MAX_HARDNAIVE)-high

#  create-submission-$(MAX_HARDNAIVE)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_HARDNAIVE)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_HARDNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_HARDNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_HARDNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_HARDNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_HARDNAIVE)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDNAIVE)-0/task1/00ALL-$*-dev.result.txt ;

###################################################################
# END MAX_HARDNAIVE
####################################################################

###################################################################
# BEGIN MAX_TRANSSMART
####################################################################
# Evaluate the effect of smart vs naive alignment
MAX_TRANSSMART:= TS

# Ensemble Members of MAX_TRANSSMART
low-$(MAX_TRANSSMART)-member    += x-strcrp-a30-o-e60-h200-b1-r0-x
medium-$(MAX_TRANSSMART)-member += x-strcrp-a20-o-e60-h200-b1-r0-x
high-$(MAX_TRANSSMART)-member   += x-strcrp-a10-o-e35-h200-b1-r0-x
low-$(MAX_TRANS)-member    += x-strcrp-a30-o-e60-h200-b1-r0-x
low-$(MAX_TRANS)-member    += x-strdmb-a30-o-e60-h200-b1-r0-x

medium-$(MAX_TRANS)-member += x-strdmb-a20-o-e60-h200-b1-r0-x

high-$(MAX_TRANS)-member   += x-strdmb-a10-o-e35-h200-b1-r0-x


# Targets of MAX_TRANSSMART
create-submission-target += create-submission-$(MAX_TRANSSMART)-low
create-submission-target += create-submission-$(MAX_TRANSSMART)-medium 
create-submission-target += create-submission-$(MAX_TRANSSMART)-high

#  create-submission-$(MAX_TRANSSMART)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_TRANSSMART)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_TRANSSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_TRANSSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_TRANSSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_TRANSSMART)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_TRANSSMART)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSSMART)-0/task1/00ALL-$*-dev.result.txt ;

###################################################################
# END MAX_TRANSSMART
####################################################################


###################################################################
# BEGIN MAX_TRANSNAIVE
####################################################################
# Evaluate the effect of smart vs naive alignment
MAX_TRANSNAIVE:= TN

# Ensemble Members of MAX_TRANSNAIVE
low-$(MAX_TRANSNAIVE)-member    += x-strdmb-a30-o-e60-h200-b1-r0-x
medium-$(MAX_TRANSNAIVE)-member += x-strdmb-a20-o-e60-h200-b1-r0-x
high-$(MAX_TRANSNAIVE)-member   += x-strdmb-a10-o-e35-h200-b1-r0-x


# Targets of MAX_TRANSNAIVE
create-submission-target += create-submission-$(MAX_TRANSNAIVE)-low
create-submission-target += create-submission-$(MAX_TRANSNAIVE)-medium 
create-submission-target += create-submission-$(MAX_TRANSNAIVE)-high

#  create-submission-$(MAX_TRANSNAIVE)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_TRANSNAIVE)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_TRANSNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_TRANSNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_TRANSNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_TRANSNAIVE)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_TRANSNAIVE)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANSNAIVE)-0/task1/00ALL-$*-dev.result.txt ;

###################################################################
# END MAX_TRANSNAIVE
####################################################################


#https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-02-ensemble-hard
ENS_HARD:=02


low-$(ENS_HARD)-member    += $(foreach r,$(LOW_MEMBERS),x-shdmix-a-o-e50-h200-b1-r$(r)-x)
low-$(ENS_HARD)-member    += $(foreach r,$(LOW_MEMBERS),x-shdmixdmb-a-o-e50-h200-b1-r$(r)-x)
medium-$(ENS_HARD)-member += $(foreach r,$(MEDIUM_MEMBERS),x-shdmix-a-o-e30-h200-b1-r$(r)-x)
medium-$(ENS_HARD)-member += $(foreach r,$(MEDIUM_MEMBERS),x-shdmixdmb-a-o-e30-h200-b1-r$(r)-x)
high-$(ENS_HARD)-member   += $(foreach r,$(HIGH_MEMBERS),x-shdmix-a-o-e20-h200-b1-r$(r)-x)
high-$(ENS_HARD)-member   += $(foreach r,$(HIGH_MEMBERS),x-shdmixdmb-a-o-e20-h200-b1-r$(r)-x)

# Targets of MAX_HARD
create-submission-target += create-submission-$(ENS_HARD)-low 
create-submission-target += create-submission-$(ENS_HARD)-medium 
create-submission-target += create-submission-$(ENS_HARD)-high

create-submission-$(ENS_HARD)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/
	printf "# STARTING SUBMISSION ENS_HARD $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(ENS_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only   --nbest $(NBEST_IND) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(ENS_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --nbest $(NBEST_IND) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(ENS_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(ENS_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(ENS_HARD)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARD)-0/task1/00ALL-$*-dev.result.txt ;



# https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-03-max-trans
MAX_TRANS:= 03

# Ensemble Members of MAX_TRANS 
low-$(MAX_TRANS)-member    += x-strcrp-a30-o-e60-h200-b1-r0-x
low-$(MAX_TRANS)-member    += x-strdmb-a30-o-e60-h200-b1-r0-x
medium-$(MAX_TRANS)-member += x-strcrp-a20-o-e60-h200-b1-r0-x
medium-$(MAX_TRANS)-member += x-strdmb-a20-o-e60-h200-b1-r0-x
high-$(MAX_TRANS)-member   += x-strcrp-a10-o-e35-h200-b1-r0-x
high-$(MAX_TRANS)-member   += x-strdmb-a10-o-e35-h200-b1-r0-x

# Targets of MAX_TRANS
create-submission-target += create-submission-$(MAX_TRANS)-low 
create-submission-target += create-submission-$(MAX_TRANS)-medium 
create-submission-target += create-submission-$(MAX_TRANS)-high


# https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-04-ensemble-trans
ENS_TRANS:= 04

low-$(ENS_TRANS)-member    += $(foreach r,$(LOW_MEMBERS),x-strcrp-a30-o-e60-h200-b1-r$(r)-x)
low-$(ENS_TRANS)-member    += $(foreach r,$(LOW_MEMBERS),x-strdmb-a30-o-e60-h200-b1-r$(r)-x)
medium-$(ENS_TRANS)-member += $(foreach r,$(MEDIUM_MEMBERS),x-strcrp-a20-o-e60-h200-b1-r$(r)-x)
medium-$(ENS_TRANS)-member += $(foreach r,$(MEDIUM_MEMBERS),x-strdmb-a20-o-e60-h200-b1-r$(r)-x)
high-$(ENS_TRANS)-member   += $(foreach r,$(HIGH_MEMBERS),x-strcrp-a10-o-e35-h200-b1-r$(r)-x)
high-$(ENS_TRANS)-member   += $(foreach r,$(HIGH_TRANS_DMB_MEMBERS),x-strdmb-a10-o-e35-h200-b1-r$(r)-x)

# Targets of MAX_HARD
create-submission-target += create-submission-$(ENS_TRANS)-low 
create-submission-target += create-submission-$(ENS_TRANS)-medium 
create-submission-target += create-submission-$(ENS_TRANS)-high



# Max between HARDTRANS Runs
# https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-05-max-hardtrans
MAX_HARDTRANS:= 05

# Ensemble members of MAX_HARDTRANS
low-$(MAX_HARDTRANS)-member    += $(low-$(MAX_TRANS)-member)    $(low-$(MAX_HARD)-member)
medium-$(MAX_HARDTRANS)-member += $(medium-$(MAX_TRANS)-member) $(medium-$(MAX_HARD)-member)
high-$(MAX_HARDTRANS)-member   += $(high-$(MAX_TRANS)-member)   $(high-$(MAX_HARD)-member)
high-$(MAX_HARDTRANS)-member   += x-snematus-a-o-e100-h600-b1-r1-x 


# Targets of MAX_HARDTRANS
create-submission-target += create-submission-$(MAX_HARDTRANS)-low 
create-submission-target += create-submission-$(MAX_HARDTRANS)-medium 
create-submission-target += create-submission-$(MAX_HARDTRANS)-high


# https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-06-ensemble-hard-trans
ENS_HARDTRANS:= 06

# Ensemble members of ENS_HARDTRANS
low-$(ENS_HARDTRANS)-member    += $(low-$(ENS_TRANS)-member)    $(low-$(ENS_HARD)-member)
medium-$(ENS_HARDTRANS)-member += $(medium-$(ENS_TRANS)-member) $(medium-$(ENS_HARD)-member)
high-$(ENS_HARDTRANS)-member   += $(high-$(ENS_TRANS)-member)   $(high-$(ENS_HARD)-member)
high-$(ENS_HARDTRANS)-member   += x-snematus-a-o-e100-h600-b1-r1-x


# Targets of ENS_HARDTRANS
create-submission-target += create-submission-$(ENS_HARDTRANS)-low 
create-submission-target += create-submission-$(ENS_HARDTRANS)-medium 
create-submission-target += create-submission-$(ENS_HARDTRANS)-high

nbest-target += create-submission-$(ENS_HARDTRANS)-low-$(NBEST)
nbest-target += create-submission-$(ENS_HARDTRANS)-medium-$(NBEST)
nbest-target += create-submission-$(ENS_HARDTRANS)-high-$(NBEST)


# https://gitlab.cl.uzh.ch/makarov/conll2017/blob/master/ensemble_strategies.md#run-07-ultimate-maxensemble-hardtrans
MAX_0506:= 07

# Ensemble members of ENS_HARDTRANS
low-$(MAX_0506)-member    += CLUZH-05-0 CLUZH-06-0-15
medium-$(MAX_0506)-member += CLUZH-05-0 CLUZH-06-0-15
high-$(MAX_0506)-member   += CLUZH-05-0 CLUZH-06-0-07 
#high-$(MAX_0506)-member   += CLUZH-10-0

# Targets of MAX_TRANS
create-recursive-submission-target += create-submission-$(MAX_0506)-low 
create-recursive-submission-target += create-submission-$(MAX_0506)-medium 
create-recursive-submission-target += create-submission-$(MAX_0506)-high


#  create-submission-$(MAX_0506)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_0506)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_0506)-member),$(SUBMISSIONDIR)/$(m)/task1/$$l-$*-out)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_0506)-member),$(SUBMISSIONDIR)/$(m)/task1/$$l-$*-out) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_0506)-member),$(SUBMISSIONDIR)/$(m)/task1/$$l-$*-out-dev) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_0506)-member),$(SUBMISSIONDIR)/$(m)/task1/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_0506)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_0506)-0/task1/00ALL-$*-dev.result.txt ;


# load all settings
include exp0.conf.mk






# Could be generalized by https://stackoverflow.com/questions/15167766/makefile-dynamic-rules-w-no-gnu-make-pattern

#  create-submission-$(MAX_HARD)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_HARD)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_HARD)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_HARD)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARD)-0/task1/00ALL-$*-dev.result.txt ;



#  create-submission-$(MAX_TRANS)-% # the pattern encodes the setting low, medium, high
create-submission-$(MAX_TRANS)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_TRANS)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_TRANS)-0/task1/00ALL-$*-dev.result.txt ;


#  create-submission-$(MAX_HARDTRANS)-% # the pattern encodes the setting low, medium, high
# Could be generalized by https://stackoverflow.com/questions/15167766/makefile-dynamic-rules-w-no-gnu-make-pattern
create-submission-$(MAX_HARDTRANS)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/
	printf "# STARTING SUBMISSION $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(MAX_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(MAX_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l --max_strategy \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(MAX_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(MAX_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(MAX_HARDTRANS)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/00ALL-$*-dev.result.txt ;
#	touch $(SUBMISSIONDIR)/$(INSTITUTION)-$(MAX_HARDTRANS)-0/task1/00ALL-$*-dev.result.txt.done




create-submission-$(ENS_TRANS)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/
	printf "# STARTING SUBMISSION ENS_TRANS $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(ENS_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only  --nbest $(NBEST_IND) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(ENS_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest) | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l   --nbest $(NBEST_IND) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(ENS_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(ENS_TRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(ENS_TRANS)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_TRANS)-0/task1/00ALL-$*-dev.result.txt ;


create-submission-$(ENS_HARDTRANS)-%: 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/
	printf "# STARTING SUBMISSION ENS_HARDTRANS $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --nbest $(NBEST) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)  | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l   --nbest $(NBEST) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(ENS_HARDTRANS)-0-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0/task1/00ALL-$*-dev.result.txt ;


create-submission-$(ENS_HARDTRANS)-%-$(NBEST): 
	mkdir -p $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/
	printf "# STARTING SUBMISSION ENS_HARDTRANS $@\n" ; \
	for l in $(LNG) ; do \
		if test -e $(word 1, $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)) ; \
		then \
			printf "\r%-40s" $$l ;\
			./ensemble_from_output_dev.py --lang $$l --test_only --nbest $(NBEST) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/$$l-$*-out \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/$$l-$*-out-dev \
				--input $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest)  | tee $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/$$l-$*-out.err ; \
			./ensemble_from_output_dev.py --lang $$l   --nbest $(NBEST) \
				--pred_out   $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/$$l-$*-out-dev \
				--result_out $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/$$l-$*-out-dev.result.txt \
				--input $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.dev.predictions.nbest) ; \
		else \
			echo "#INFO: Ignoring $$l; missing file $(word 1, $(foreach m,$($*-$(ENS_HARDTRANS)-member),$(RESULTSDIR)/$(m)/$$l-$*.best.test.test.predictions.nbest))" ; \
		fi \
	done
	printf  "%s\n" $(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)-$* > $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/00ALL-$*-dev.result.txt ; for l in $(LNG) ; do if test -e $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/$$l-$*-out-dev.result.txt ; then printf "%1.3f\n" $$(awk '/Accuracy/ {print $$4}' $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/$$l-$*-out-dev.result.txt  ) ; else printf "n/a\n" ; fi ; done >> $(SUBMISSIONDIR)/$(INSTITUTION)-$(ENS_HARDTRANS)-0-$(NBEST)/task1/00ALL-$*-dev.result.txt ;


create-dev-diffs:
	#
	# CREATING diffs for dev sets
	for d in $$(ls -d  $(RESULTSDIR)/x-*-x ) ; do \
		for l in $(LNG) ; do \
			if test -e $$d/$$l-$(SETTING).best.dev.predictions.nbest ; then \
				paste ../data/all/task1/$${l}-dev $$d/$$l-$(SETTING).best.dev.predictions.nbest |gawk -v FS=$$'\t' -v OFS=$$'\t'  ' $$5 != $$2 { print $$1,$$2,$$3,$$5 } '  > $$d/$$l-$(SETTING).best.dev.predictions.nbest.diff.txt ;\
			fi \
		done \
	done
create-submission-diffs:
	#
	# CREATING diffs for dev sets
	for d in $$(ls -d  $(SUBMISSIONDIR)/$(INSTITUTION)* ) ; do \
		for l in $(LNG) ; do \
				paste ../data/all/task1/$${l}-dev $$d/task1/$$l-$(SETTING)-out-dev |gawk -v FS=$$'\t' -v OFS=$$'\t'  ' $$5 != $$2 { print $$1,$$2,$$3,$$5 } '  > $$d/task1/$$l-$(SETTING)-out-dev.diff.txt ;\
		done \
	done


create-dev-stats:
	paste $(shell ls $(SUBMISSIONDIR)/*/task1/00ALL-low-dev.result.txt) > $(SUBMISSIONDIR)/00ALL-low-dev.result.txt
	paste $(shell ls $(SUBMISSIONDIR)/*/task1/00ALL-medium-dev.result.txt) > $(SUBMISSIONDIR)/00ALL-medium-dev.result.txt
	paste $(shell ls $(SUBMISSIONDIR)/*/task1/00ALL-high-dev.result.txt) > $(SUBMISSIONDIR)/00ALL-high-dev.result.txt



create-nbest: $(nbest-target)

create-nbest-selection:
	for NB in 05 07 09 11 15 ; do NBEST=$${NB} make -f submission.mk create-nbest -j 4 & done 

create-submissions : $(create-submission-target)

create-all-recursive-submissions:
	make -f submission.mk $(create-recursive-submission-target)
create-all-submissions: 
	make -f submission.mk $(create-submission-target) -j 24
	for NB in 05 07 09 11 15 ; do NBEST=$${NB} make -f submission.mk create-nbest -j 4 & done ; wait
	make -f submission.mk $(create-recursive-submission-target)
	make -f submission.mk  inject-06-submission

inject-06-submission:
	cp -v $(SUBMISSIONDIR)/$(INSTITUTION)-06-0-15/task1/*-low-out* $(SUBMISSIONDIR)/$(INSTITUTION)-06-0-15/task1/*-medium-out*  $(SUBMISSIONDIR)/$(INSTITUTION)-06-0/task1/
	cp -v $(SUBMISSIONDIR)/$(INSTITUTION)-06-0-07/task1/*-high-out* $(SUBMISSIONDIR)/$(INSTITUTION)-06-0/task1/

create-zips:
	cd $(SUBMISSIONDIR)  ; \
	for d in $(INSTITUTION)-*-0 ; do \
	 	echo $$d ; \
	 	zip -ur $$d.zip $$d -i $$d/*-out ; \
	done


# conll2017/evaluation/evalm.py --task 1 --guess $< --gold exp0.d/data/d-a1-d/$*-dev.id > $@ 2> $@.err

create-evaluations:
	make -f submission.mk $(SUBMISSIONDIR)/create-evaluation-low.done $(SUBMISSIONDIR)/create-evaluation-medium.done $(SUBMISSIONDIR)/create-evaluation-high.done -j 10
	make -f submission.mk create-evaluation-summary-low create-evaluation-summary-medium create-evaluation-summary-high 
$(SUBMISSIONDIR)/create-evaluation-%.done:
	cd $(SUBMISSIONDIR) && \
	for d in $$( find . -type d -name "$(INSTITUTION)*" ) ; do \
		export shortd=$${d%%/$(INSTITUTION)} ; \
		printf "$${shortd}-%s\t$${shortd}-%s\n" ACC LEV > $$d/$*.evalm.tsv  ; \
		for l in $(LNG) ; do \
			if test -e $${d}/task1/$$l-$*-out-dev ; \
			then \
				../src/evalm.py --lang $$l --task 1 --guess $${d}/task1/$$l-$*-out-dev --gold ../src/$(GOLDDIR)/$$l-dev | cut -f 2- >> $$d/$*.evalm.tsv ; \
			else \
				printf "n/a\tn/a\n" >> $$d/$*.evalm.tsv ; \
			fi ;\
		done & \
	done ; \
	wait
	touch $@

create-evaluation-summary-%:
	cd $(SUBMISSIONDIR) && \
	paste $$(ls $(INSTITUTION)*/$*.evalm.tsv) > $*.evalm.tsv

PAPER-DEV-SYSTEMS+=CLUZH-HN-0
PAPER-DEV-SYSTEMS+=CLUZH-HS-0
PAPER-DEV-SYSTEMS+=CLUZH-TN-0
PAPER-DEV-SYSTEMS+=CLUZH-TS-0
PAPER-DEV-SYSTEMS+=CLUZH-01-0
PAPER-DEV-SYSTEMS+=CLUZH-02-0
PAPER-DEV-SYSTEMS+=CLUZH-03-0
PAPER-DEV-SYSTEMS+=CLUZH-04-0
PAPER-DEV-SYSTEMS+=CLUZH-05-0
PAPER-DEV-SYSTEMS+=CLUZH-06-0
PAPER-DEV-SYSTEMS+=CLUZH-07-0
PAPER-DEV-SYSTEMS+=CLUZH-BASELINE
create-paper-evaluation-stats:
	paste $(foreach s,$(PAPER-DEV-SYSTEMS), $(SUBMISSIONDIR)/$s/low.evalm.tsv) > $(SUBMISSIONDIR)/00PAPER-low.evalm.tsv
	paste $(foreach s,$(PAPER-DEV-SYSTEMS), $(SUBMISSIONDIR)/$s/medium.evalm.tsv) > $(SUBMISSIONDIR)/00PAPER-medium.evalm.tsv
	paste $(foreach s,$(PAPER-DEV-SYSTEMS), $(SUBMISSIONDIR)/$s/high.evalm.tsv) > $(SUBMISSIONDIR)/00PAPER-high.evalm.tsv

SHELL:=/bin/bash -x
