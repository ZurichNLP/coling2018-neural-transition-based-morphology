start_time1=`date +%s`
python run_all_langs_generic.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=100 --layers=1 --optimization=ADADELTA  --pool=25  --script=hard_attention.py --prefix=HARD --regimes=medium,low
end_time1=`date +%s`

start_time2=`date +%s`
python run_all_langs_generic.py --dynet-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=100 --layers=1 --optimization=ADADELTA  --pool=25  --script=hard_attention.py --prefix=DUMB --regimes=medium,low  --params=align_dumb
end_time2=`date +%s`

start_time3=`date +%s`
python run_all_langs_generic.py --dynet-mem 4096 --input=300 --hidden=100 --feat-input=300 --epochs=100 --layers=1 --optimization=ADADELTA  --pool=25  --script=soft_attention.py --prefix=SOFT --regimes=medium,low
end_time3=`date +%s`


echo execution time HARD was `expr $end_time1 - $start_time1` s.
echo execution time DUMB was `expr $end_time2 - $start_time2` s.
echo execution time SOFT was `expr $end_time3 - $start_time3` s.
