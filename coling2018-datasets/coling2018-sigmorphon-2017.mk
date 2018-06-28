# targets to extract the test results from the experiments for the official repository with normalized file naming conventions


create-links-sgm2017-low:
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acls-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ca-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acls-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ca-lcs-e
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acrp-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ca-crp-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acrp-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ca-crp-e
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acls-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ca-mrt-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acls-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ca-mrt-lcs-e
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acrp-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ca-mrt-crp-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-haem-acrp-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ca-mrt-crp-e
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acls-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ha-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acls-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ha-lcs-e
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acrp-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ha-crp-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acrp-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x      sgm2017-low-ha-crp-e
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acrp-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ha-mrt-crp-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acrp-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ha-mrt-crp-e
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acls-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ha-mrt-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017low/x-hard-acls-psgm2017low-n200_1-w100_20_100_T-e60_20-oADADELTA_0-mmrt1-x     sgm2017-low-ha-mrt-lcs-e

sgm2017-low-zip-files := \
  sgm2017-low-ca-lcs-avg.zip \
  sgm2017-low-ca-lcs-e.zip \
  sgm2017-low-ca-crp-avg.zip \
  sgm2017-low-ca-crp-e.zip \
  sgm2017-low-ca-mrt-lcs-avg.zip \
  sgm2017-low-ca-mrt-lcs-e.zip \
  sgm2017-low-ca-mrt-crp-avg.zip \
  sgm2017-low-ca-mrt-crp-e.zip \
  sgm2017-low-ha-lcs-avg.zip \
  sgm2017-low-ha-lcs-e.zip \
  sgm2017-low-ha-crp-avg.zip \
  sgm2017-low-ha-crp-e.zip \
  sgm2017-low-ha-mrt-crp-avg.zip \
  sgm2017-low-ha-mrt-crp-e.zip \
  sgm2017-low-ha-mrt-lcs-avg.zip \
  sgm2017-low-ha-mrt-lcs-e.zip \
#	

sgm2017-low-zip-target: create-links-sgm2017-low $(sgm2017-low-zip-files)

	
create-links-sgm2017-medium:
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acls-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ca-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acls-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ca-lcs-e
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ca-crp-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ca-crp-e
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acls-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ca-mrt-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acls-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ca-mrt-lcs-e
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ca-mrt-crp-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-haem-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ca-mrt-crp-e
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acls-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ha-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acls-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ha-lcs-e
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ha-crp-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_10-oADADELTA_0-mmle-x      sgm2017-medium-ha-crp-e
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ha-mrt-crp-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acrp-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ha-mrt-crp-e 
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acls-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ha-mrt-lcs-avg
	ln -fvs ../../paper2018/results/sgm2017medium/x-hard-acls-psgm2017medium-n200_1-w100_20_100_T-e50_15-oADADELTA_0-mmrt1-x     sgm2017-medium-ha-mrt-lcs-e

sgm2017-medium-zip-files := \
  sgm2017-medium-ca-lcs-avg.zip \
  sgm2017-medium-ca-lcs-e.zip \
  sgm2017-medium-ca-crp-avg.zip \
  sgm2017-medium-ca-crp-e.zip \
  sgm2017-medium-ca-mrt-lcs-avg.zip \
  sgm2017-medium-ca-mrt-lcs-e.zip \
  sgm2017-medium-ca-mrt-crp-avg.zip \
  sgm2017-medium-ca-mrt-crp-e.zip \
  sgm2017-medium-ha-lcs-avg.zip \
  sgm2017-medium-ha-lcs-e.zip \
  sgm2017-medium-ha-crp-avg.zip \
  sgm2017-medium-ha-crp-e.zip \
  sgm2017-medium-ha-mrt-crp-avg.zip \
  sgm2017-medium-ha-mrt-crp-e.zip \
  sgm2017-medium-ha-mrt-lcs-avg.zip \
  sgm2017-medium-ha-mrt-lcs-e.zip \
#	

sgm2017-medium-zip-target: create-links-sgm2017-medium $(sgm2017-medium-zip-files)


%-e.zip:
	zip $@ $(@:.zip=)/*_/s_0/f.beam4.test.predictions

%-avg.zip:
	zip $@ $$(ls $(@:.zip=)/*_/s_{1,2,3,4,5}/f.beam4.test.predictions )



SHELL:=/bin/bash
