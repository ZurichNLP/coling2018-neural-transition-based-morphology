# targets to extract the test results from the experiments for the official repository with normalized file naming conventions
# only the models with random initialization seed 5 are used (not selected according to any optimality)

create-links-lemmatizationwic2002:
	ln -fvs ../../paper2018/results/lemmatization/x-hard-acrp-plemmatization-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x    lemmatizationwic2002-ha-crp-avg
	ln -fvs ../../paper2018/results/lemmatization/x-haem-acrp-plemmatization-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x    lemmatizationwic2002-ca-crp-avg
	ln -fvs ../../paper2018/results/lemmatization/x-hard-acls-plemmatization-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x    lemmatizationwic2002-ha-lcs-avg
	ln -fvs ../../paper2018/results/lemmatization/x-haem-acls-plemmatization-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x    lemmatizationwic2002-ca-lcs-avg


lemmatizationwic2002-zip-files := \
  lemmatizationwic2002-ha-crp-avg.zip \
  lemmatizationwic2002-ca-crp-avg.zip \
  lemmatizationwic2002-ha-lcs-avg.zip \
  lemmatizationwic2002-ca-lcs-avg.zip \


lemmatizationwic2002-zip-target: create-links-lemmatizationwic2002 $(lemmatizationwic2002-zip-files)


%-avg.zip:
	zip $@ $$(ls $(@:.zip=)/*_?_/s_5/f.beam4.test.predictions )



SHELL:=/bin/bash
