all: libalign.so

libalign.so: align.c
	gcc -O3 -Wall -Wextra -shared -fPIC align.c -o libalign.so

clean:
	/bin/rm libalign.so *.pyc


EXP_MAKEFILE?=exp0.conf.mk
###### Convenience targets for experimentation


make-dev-stats:
	for SETTING in low medium high ; do SETTING=$${SETTING} $(MAKE) -f $(EXP_MAKEFILE) $@ ; done
