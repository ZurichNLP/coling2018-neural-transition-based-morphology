# Research code for the coling 2018 paper "Neural Transition-based String Transduction for Limited-Resource Setting in Morphology" by Peter Makarov and Simon Clematide 
This [code basis](https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology) allows for the reproduction of the results of our experiments.
Note that there is a separate git repository that contains all our [test data results of our reported systems](https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology-test-data) in a consistently named and organized way. 
And it also contains the respective gold standard data.

The [paper](paper/coling-2018-paper.pdf) is in the repository as well.

## Requirements
Our environment when running the experiments:
 - Debian 9
 - python 2.7.13
 - [dynet v2.0.2](https://github.com/clab/dynet/releases/tag/2.0.2)
 - a few external python modules: docopt progressbar editdistance 

All our experiments were run on CPU, which can take some hours for larger datasets.

### Setup
```git clone --recursive ```

In order to setup the Chinese Restaurant Character Aligner binaries:

```make setup```



## Computing models
There is a configuration file for each dataset. The easiest way is to generate a shell script that contains all commands for training the models (note that the shell commands are not executed by the launcher by default).
All results are stored in a parent folder of this repository `../paper2018/results`.
All models are automatically evaluated.

### Sigmorphon 2017 low

``` cd lib && python  ./launch_dataset.py ../dataset_configs/sgm2017low_config.py > sgm2017low.sh ; bash -x sgm2017low.sh ``` 

The test set results for a language LANG and seed SEED can be found in ` ../paper2018/results/sgm2017low/x-haem-acls-psgm2017low-n200_1-w100_20_100_T-e60_15-oADADELTA_0-mmle-x/{LANG}_/s_{SEED}}/f.beam4.test.predictions`.
The relevant experimental variables are encoded in the directory below the shortname of the data set (e.g. sgm2017low) as follows:
 - `-haem` = CA
 - `-hard` = HA\*
 - `-acls` = LCS (longest common substring alignment)
 - `-acrp` = CRP (Chinese restaurant process)
 - `-n200_1` = One hidden layer of dimension 200
 - `-e60_15` = 60 epochs with an early stopping patience of 15
 - `-oADADELTA` = ADADELTA optimizer
 - `-mmle` = Maximum Likelihood Estimation



### Sigmorphon 2017 medium

``` cd lib && python launch_dataset.py ../dataset_configs/sgm2017medium_config.py > sgm2017medium.sh ; bash -x sgm2017medium.sh  ``` 

### Sigmorphon 2016

``` cd lib && python launch_dataset.py ../dataset_configs/sigmorphon2016_config.py > sgm2016low.sh ; bash -x sgm2016low.sh```


### Lemmatization

``` cd lib && python launch_dataset.py ../dataset_configs/sigmorphon2016_config.py > sgm2016low.sh ; bash -x sgm2016low.sh```


### Celex all
``` cd lib && python launch_dataset.py ../dataset_configs/celex_config.py > celexall.sh ; bash -x celexall.sh```

### Celex by task

``` cd lib && 
	for task in 2PIE 2PKE 13SIA rP ; do 	
		python launch_dataset.py ../dataset_configs/celex${task}_config.py > celexbytask${task}.sh ; bash -x  celexbytask${task}.sh ;
	done
```


## Computing ensembles
The ensembling of models and their evaluation works analogue to the model launching:

```cd lib && python launch_dataset_ensembling.py ../dataset_configs/sgm2017low_config.py```


## Connection between official results and computed results
The exact definition of which results from which result directory actually went into which table of our paper (and the corresponding test data in <https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology-test-data>) can be found 
in the makefiles in <https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology/coling2018-datasets> and in <https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology-test-data>.  


Feel free to ask <simon.clematide@uzh.ch> for questions related to the data or <makarov@cl.uzh.ch> for questions regarding the code.

We want to thank Tatyana Ruzsics for all her valuable input, code and inspiration!


