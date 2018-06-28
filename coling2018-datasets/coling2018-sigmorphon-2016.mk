# targets to extract the test results from the experiments for the official repository with normalized file naming conventions


create-links-sgm2016:
	ln -fvs ../../paper2018/results/sigmorphon2016/x-hard-acrp-psigmorphon2016-n200_1-w100_20_100_T-e30_5-oADADELTA_0-mmle-x     sgm2016-ha-crp-avg
	ln -fvs ../../paper2018/results/sigmorphon2016/x-haem-acrp-psigmorphon2016-n200_1-w100_20_100_T-e30_5-oADADELTA_0-mmle-x     sgm2016-ca-crp-avg
	ln -fvs ../../paper2018/results/sigmorphon2016/x-hard-acrp-psigmorphon2016-n200_1-w100_20_100_T-e30_5-oADADELTA_0-mmle-x     sgm2016-ha-crp-e
	ln -fvs ../../paper2018/results/sigmorphon2016/x-haem-acrp-psigmorphon2016-n200_1-w100_20_100_T-e30_5-oADADELTA_0-mmle-x     sgm2016-ca-crp-e

sgm2016-zip-files := \
  sgm2016-ha-crp-avg.zip \
  sgm2016-ca-crp-avg.zip \
  sgm2016-ha-crp-e.zip \
  sgm2016-ca-crp-e.zip \

sgm2016-zip-target: create-links-sgm2016 $(sgm2016-zip-files)

	


%-e.zip:
	zip $@ $(@:.zip=)/*task1/s_0/f.beam4.test.predictions

%-avg.zip:
	zip $@ $$(ls $(@:.zip=)/*task1/s_{1,2,3,4,5}/f.beam4.test.predictions )



SHELL:=/bin/bash
