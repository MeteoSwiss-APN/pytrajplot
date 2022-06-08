#!/bin/bash -l
# Option -l needed to run as cron job

# Settings
# --------
# Conda environment
conda_env=pytrajplot
# Lagranto-output storage location
store_osm=/store/mch/msopr/osm
# pytrajplot output location
pytrajplot_out=local
# Graphics format
datatype_opt="--datatype png --datatype pdf"
# Domain options
domain_opt_c1="--domain ch --domain alps"
domain_opt_ie="--domain alps --domain centraleurope --domain europe"
domain_opt_ig="--domain dynamic --domain dynamic_zoom"
# --------

# Yesterday 06 UTC
bt_06=$(date --utc --date="yesterday 06" +%Y%m%d%H)
# Yesterday 12 UTC
bt_12=$(date --utc --date="yesterday 12" +%Y%m%d%H)
# Yesterday 18 UTC
bt_18=$(date --utc --date="yesterday 18" +%Y%m%d%H)
# Yesterday 21 UTC
bt_21=$(date --utc --date="yesterday 21" +%Y%m%d%H)
# Today 00 UTC
bt_00=$(date --utc --date="today 00" +%Y%m%d%H)
# Today 03 UTC
bt_03=$(date --utc --date="today 03" +%Y%m%d%H)

# Load conda env if CONDA_PREFIX not defined
[[ -z $CONDA_PREFIX ]] && conda activate $conda_env

for basetime in $bt_06 $bt_12 $bt_18 $bt_21 $bt_00 $bt_03 ; do
    yy=${basetime:2:2}
    yymmddhh=${basetime:2}
    hh=${basetime:8:2}

    echo "*****"
    echo Basetime: $(date --utc --date="${basetime:0:8} $hh" "+%F %H UTC")

    # Operational INPUT_DIRs:
    input_dir_c1=$(echo $store_osm/COSMO-1E/FCST${yy}/${yymmddhh}_4??/lagranto_c/000)
    input_dir_ie=$store_osm/IFS-HRES/IFS-HRES-LAGRANTO${yy}/${yymmddhh}_LIH/lagranto_f
    input_dir_ig=$store_osm/IFS-HRES/IFS-HRES-LAGRANTO${yy}/${yymmddhh}_LIH/lagranto_c

    # Output directories
    output_dir_c1=$pytrajplot_out/plot_c1_${basetime}
    output_dir_ie=$pytrajplot_out/plot_ie_${basetime}
    output_dir_ig=$pytrajplot_out/plot_ig_${basetime}

    # Submit jobs
    for model in c1 ie ig ; do
        eval input_dir=\$input_dir_$model
        eval output_dir=\$output_dir_$model
        eval domain_opt=\$domain_opt_$model

        if [[ -r $input_dir ]] ; then
            echo Input for $model: $(ls -d $input_dir)
            [[ ! -d $output_dir ]] && mkdir -p $output_dir
            echo Output for $model: $(ls -d $output_dir)
            batchPP -n ${model}_$hh -- \
                $CONDA_PREFIX/bin/pytrajplot $input_dir $output_dir $datatype_opt $domain_opt
        else
            echo "No data for $model, skipping: $input_dir"
        fi
    done
done
