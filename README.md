# Research code for the coling 2018 paper "" by Peter Makarov and Simon Clematide 
This code basis allows for the reproduction of the results of our experiments.
Note that there is a separate git repository that contains all our test data results of our reported systems (https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology-test-data) in a consistently organized way.


## Requirements
Our environment when running the experiments:
 - Debian 9
 - python 2.7.13
 - dynet v2.0.2
 - a few external python modules: docopt progressbar editdistance 

All our experiments were run on CPU, which can take some hours for larger datasets.

### Setup
```git clone --recursive ```

In order to setup a few things and compile the Chinese Restaurant Character Aligner

```make setup```



## Computing models
There is a configuration file for each dataset. The easiest way is to generate a shell script that contains all commands for training the models. 
All results are stored in a parent folder of this repository `../paper2018/results`.
All models are automatically evaluated.

### Sigmorphon 2017 low

``` cd lib && python  ./launch_dataset.py ../dataset_configs/sgm2017low_config.py > sgm2017low.sh ; bash -x sgm2017low.sh ``` 

### Sigmorphon 2017 medium

``` cd lib && python  ./launch_dataset.py ../dataset_configs/sgm2017medium_config.py > sgm2017medium.sh ; bash -x sgm2017medium.sh  ``` 

### Sigmorphon 2016

``` cd lib && python  ./launch_dataset.py ../dataset_configs/sigmorphon2016_config.py > sgm2016low.sh ; bash -x sgm2016low.sh```


### Lemmatization

``` cd lib && python  ./launch_dataset.py ../dataset_configs/sigmorphon2016_config.py > sgm2016low.sh ; bash -x sgm2016low.sh```

## Computing ensembles
The ensembling of models and their evaluation works similar to the model creation.

```cd lib && python ```
