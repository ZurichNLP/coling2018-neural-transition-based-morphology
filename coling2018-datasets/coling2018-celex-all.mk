# targets to extract the test results from the experiments for the official repository with normalized file naming conventions


create-links-celexall-low:
	ln -fvs ../../paper2018/results/newcelex/x-haem-acrp-pcelex-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      celexall-ca-crp-e
	ln -fvs ../../paper2018/results/newcelex/x-haem-acrp-pcelex-n200_1-w100_20_100_T-e50_20-oADADELTA_0-mmrt1-x     celexall-ca-mrt-crp-e
	ln -fvs ../../paper2018/results/newcelex/x-hard-acrp-pcelex-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      celexall-ha-crp-e
	ln -fvs  ../../paper2018/results/newcelex/x-hard-acrp-pcelex-n200_1-w100_20_100_T-e50_20-oADADELTA_0-mmrt1-x    celexall-ha-mrt-crp-e


celexall-zip-files := \
  celexall-ca-crp-e.zip \
  celexall-ha-crp-e.zip \
  celexall-ca-mrt-crp-e.zip \
  celexall-ha-mrt-crp-e.zip \
#	

celexall-zip-target: create-links-celexall-low $(celexall-zip-files)

# Folds are numbered from 0..4
# 13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_1.
%-e.zip:
	zip $@ $(@:.zip=)/13SIA-13SKE_2PIE-13PKE_2PKE-z_rP-pA_?./s_0/f.beam4.test.predictions




SHELL:=/bin/bash
