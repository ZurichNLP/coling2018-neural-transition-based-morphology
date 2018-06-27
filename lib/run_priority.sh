
#export folder_name="../priority_calls"
export folder_name="../next_priority_calls"

mkdir $folder_name

# celex
./launch_dataset.py celex_config.py  s3it_cmd_config.py -m e > $folder_name/celex-mle.sh
./launch_dataset.py celex_config.py  s3it_cmd_config.py -m r > $folder_name/celex-mrl.sh

# sigm2017low
./launch_dataset.py sgm2017low_config.py  s3it_cmd_config.py -m e > $folder_name/sigm2017low-mle.sh
./launch_dataset.py sgm2017low_config.py  s3it_cmd_config.py -m r >  $folder_name/sigm2017low-mrl.sh

# sgm2017medium
./launch_dataset.py sgm2017medium_config.py  s3it_cmd_config.py -m e >  $folder_name/sigm2017medium-mle.sh
./launch_dataset.py sgm2017medium_config.py  s3it_cmd_config.py -m r >  $folder_name/sigm2017medium-mrl.sh

# sigmorphon2016
./launch_dataset.py sigmorphon2016_config.py  s3it_cmd_config.py -m e >  $folder_name/sigm2016-mle.sh
./launch_dataset.py sigmorphon2016_config.py  s3it_cmd_config.py -m r >  $folder_name/sigm2016-mrl.sh

# segmentation
./launch_dataset.py segmentation_config.py  s3it_cmd_config.py -m e >  $folder_name/segmentation-mle.sh
./launch_dataset.py segmentation_config.py  s3it_cmd_config.py -m r >  $folder_name/segmentation-mrl.sh

# normalization
./launch_dataset.py normalisation_config.py  s3it_cmd_config.py -m e >  $folder_name/normalisation-mle.sh
./launch_dataset.py normalisation_config.py  s3it_cmd_config.py -m r >  $folder_name/normalisation-mrl.sh

# check
wc -l $folder_name/*sh
