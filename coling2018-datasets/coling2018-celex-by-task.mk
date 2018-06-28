# targets to extract the test results from the experiments for the official repository with normalized file naming conventions
subtasks := 13SIA  2PIE 2PKE  rP

create-links-celexbytask:
	$(foreach st,$(subtasks), ln -fvs ../../paper2018/results/celex$(st)/x-haem-acrp-pcelex$(st)-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x/     celexbytask-$(st)-ca-crp-avg ; )
	$(foreach st,$(subtasks), ln -fvs ../../paper2018/results/celex$(st)/x-hard-acrp-pcelex$(st)-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x/     celexbytask-$(st)-ha-crp-avg ; )
	$(foreach st,$(subtasks), ln -fvs ../../paper2018/results/celex$(st)/x-haem-acrp-pcelex$(st)-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x/     celexbytask-$(st)-ca-mrt-crp-avg ; )
	$(foreach st,$(subtasks), ln -fvs ../../paper2018/results/celex$(st)/x-hard-acrp-pcelex$(st)-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x/     celexbytask-$(st)-ha-mrt-crp-avg ; )

celexbytask-zip-files := \
  $(foreach st,$(subtasks),celexbytask-$(st)-ca-crp-avg.zip) \
  $(foreach st,$(subtasks),celexbytask-$(st)-ha-crp-avg.zip) \
  $(foreach st,$(subtasks),celexbytask-$(st)-ca-mrt-crp-avg.zip) \
  $(foreach st,$(subtasks),celexbytask-$(st)-ha-mrt-crp-avg.zip) 


celexbytask-zip-target: create-links-celexbytask $(celexbytask-zip-files)

	


# celexbytask-2PKE-ca-mrt-crp-avg/0/s_1/f.beam4.test.predictions

%-avg.zip:
	zip $@ $$(ls $(@:.zip=)/?/s_{1,2,3,4,5}/f.beam4.test.predictions )



SHELL:=/bin/bash
