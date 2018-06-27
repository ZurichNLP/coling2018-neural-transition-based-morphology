## ~/.ssh/config on your laptop
Set you own config to tunnel automatically for *.cli hosts
```
Host *.cli
	ProxyCommand ssh  <IFILDAP>@login.cl.uzh.ch -W %h:%p

```
## How can I look at the dev set performance
```
$ cd /mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/results/
$ grep 'Accuracy' x-*/*best.dev
```
## How can I generate a tab-separated file of all results (not yet computed rendered as n/a)
There is call for each setting. The file name with the output is shown after executing the command. Its content can be pasted into the google document.
```
cd /mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/src
SETTING=low make -f exp0.conf.mk make-dev-stats
SETTING=medium make -f exp0.conf.mk make-dev-stats
SETTING=high make -f exp0.conf.mk make-dev-stats
```
Or in one command (can be done from one machine):
```
for SETTING in low medium high ; do SETTING=${SETTING} make -f exp0.conf.mk make-dev-stats & done
```

## How can I produce all ensemble ouputs (r0) systems for test and dev
```
for SETTING in low medium high ; do SETTING=${SETTING} make -f exp0.conf.mk make-ensembles & done
for SETTING in low medium high ; do SETTING=${SETTING} make -f exp0.conf.mk make-ensembles-dev & done
```
## How to build all ensemble systems and update the stats
```
for SETTING in low medium high ; do SETTING=${SETTING} make -f exp0.conf.mk make-ensembles-test make-ensembles-dev make-all-ensembles-dev-r00 make-dev-stats & done
```

## How can I look at the test set output
```
$ cd /mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/results/
$ tail -n 20 x-*/*-*.best.test.test.predictions.nbest | more 
$ tail -n 20 x-*/german-low.best.test.test.predictions.nbest | more
```

## How to run all configured experiments
Not so easy to do it directly from laptop: https://stackoverflow.com/questions/29142/getting-ssh-to-execute-a-command-in-the-background-on-target-machine
Either login on each machine (with screen sessions):
````
ssh vigrid.cli  
$ pgrep -alf -- "--dynet-mem"   # check for running processes; don't overload the server ; each 
# if necessary stop all processes
$ pkill -f -- "--dynet-mem"    # make should remove the <LANG>-medium.best.dev file if interrupted (the other files will stay around)
$ cd /mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/src && touch mk/*meta && SETTING=high nohup time nice make -f exp0.conf.mk cv-x-target -j 48   2>&1 > nohup.vigrid.out &

ssh  idavoll.cli 
$ pgrep -alf -- "--dynet-mem"   # check for running processes; don't overload the server ; each 
# if necessary stop all processes
$ cd /mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/src && touch mk/*meta && SETTING=medium nohup time nice make -f exp0.conf.mk cv-x-target -j 32  2>&1  > nohup.idavoll.out &

$ pgrep -alf -- "--dynet-mem"   # check for running processes; don't overload the server ; each 
# if necessary stop all processes
ssh  gimli.cli 
$ cd /mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/src && touch mk/*meta && SETTING=low nohup time nice make -f exp0.conf.mk cv-x-target -j 32  2>&1 > nohup.gimli.out &
```
You can close the terminal afterwards (do not CTL-C it) or use screen. If you want to test what would be done, use the flag -n
```
touch mk/*meta & SETTING=??? make -f exp0.conf.mk cv-x-target -n
```
In case the make process does not have all recipes, you can force the makeing by using the option -k.

## How can I add an additional hyperparameter setting or remove an obsolete one?
Edit the file `/mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/src/exp0.conf.mk` and write new += lines (or comment out by # for documentation purposes).
If you want to overwrite all existing values, just use the operator := instead of +=
```
XV5SET += -h100
ifeq ($(SETTING),low)
XV5SET := -h50
# only XVSET with value -h50
endif
ifeq ($(SETTING),medium)
XV5SET += -h200
endif
ifeq ($(SETTING),high)
XV5SET += -h200
XV5SET += -h300
# also have 300 nodes for high setting
endif
```
The next time you start the processing, it will use this settings (not producing the commented ones)

## How can I add a new make rule template for a new system (XV1SET)
Edit the file `/mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/src/exp0.rules.mk.meta`. This templates are automatically filled with XVNSET variables and the actual rules that are included can be found here:
`/mnt/storage/bender/projects/cl/sigmorphon-uzh/conll2017/src/exp0.d/mk.d/exp0.Rules.mk`
Be carefull to type in exactly what is needed. There are 3 types of variables:
 - plain Makefile variables $(VARNAME)
 - templates for experimental variables which are expanded to the full name `_XV5_` => -h200 and -h300 (this includes the hyphen and the first character)
    - {XV5} which expands without the hyphen and first character (can often directly be used as an option argument):  `{XV5}` => 200 and 300


## How to stop all running experiments on the servers
-f option should be some uniq pattern for the processes to kill (you can only kill your processes anyway)
````
# for all servers # we identify our processes by the dynet-mem flag in the command'
for s in vigrid.cli idavoll.cli gimli.cli ; do ssh $s pkill -f -- '--dynet-mem' ; done

# work on only one sever
ssh vigrid.cli pkill -f -- '--dynet-mem'
ssh idavoll.cli pkill -f -- '--dynet-mem'
ssh gimli.cli pkill -f -- '--dynet-mem'
```