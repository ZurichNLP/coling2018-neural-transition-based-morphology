# targets to extract the test results from the experiments for the official repository with normalized file naming conventions
# ../../paper2018/results/celex13SIA0100
datasets := 13SIA   2PKE
size := 0050 0100 0300

subtasks := $(foreach ds,$(datasets),$(foreach sz,$(size),$(ds)$(sz)))


create-links-celexbytasklow:
	$(foreach st,$(subtasks), ln -fvs ../../paper2018/results/celex$(st)/x-haem-acrp-pcelex$(st)-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x/     celexbytasklow-$(st)-ca-crp-avg ; )
	$(foreach st,$(subtasks), ln -fvs ../../paper2018/results/celex$(st)/x-hard-acrp-pcelex$(st)-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x/     celexbytasklow-$(st)-ha-crp-avg ; )


celexbytasklow-zip-files := \
  $(foreach st,$(subtasks),celexbytasklow-$(st)-ca-crp-avg.zip) \
  $(foreach st,$(subtasks),celexbytasklow-$(st)-ha-crp-avg.zip) \



celexbytasklow-zip-target: create-links-celexbytasklow $(celexbytasklow-zip-files)

	


# celexbytasklow-2PKE-ca-mrt-crp-avg/0/s_1/f.beam4.test.predictions
# only 3 Models here?
%-avg.zip:
	zip $@ $$(ls $(@:.zip=)/?/s_{1,2,3}/f.beam4.test.predictions )



SHELL:=/bin/bash
