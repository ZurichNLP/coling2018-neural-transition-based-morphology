start_time1=`date +%s`
python run_all_langs_generic.py --script=hard_attention.py --prefix=RETRAIN --epochs=30 --regimes=low --langs=russsian,english,scottish-gaelic --params pretrain_path=copy_data --pool=6

python run_all_langs_generic.py --script=hard_attention.py --prefix=RETRAIN --epochs=50 --regimes=low --langs=russsian,english,scottish-gaelic --params reload --pool=6
end_time1=`date +%s`

echo execution time RETRAIN low was `expr $end_time1 - $start_time1` s.

start_time1=`date +%s`
python run_all_langs_generic.py --script=hard_attention.py --prefix=RETRAIN --epochs=10 --regimes=medium --langs=russsian,english,scottish-gaelic --params pretrain_path=copy_data --pool=6

python run_all_langs_generic.py --script=hard_attention.py --prefix=RETRAIN --epochs=50 --regimes=medium --langs=russsian,english,scottish-gaelic --params reload --pool=6
end_time1=`date +%s`

echo execution time RETRAIN medium was `expr $end_time1 - $start_time1` s.
