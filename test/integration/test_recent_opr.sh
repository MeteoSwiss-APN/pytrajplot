#!/bin/bash -l
# Option -l needed to run as cron job

# Settings
# --------
# Conda default environment
conda_env=pytrajplot
# Lagranto-output storage location
store_osm=/store_new/mch/msopr/osm
# pytrajplot output location
pytrajplot_out=local
# Graphics format
datatype_opt="--datatype png" # --datatype pdf"
# Domain options for
# ICON-CH1-EPS Control Run (icon1), IFS-HRES-Europe (ifs_e), IFS-HRES global (ifs_g)
domain_opt_icon1="--domain ch --domain alps"
domain_opt_ifs_e="--domain alps --domain centraleurope --domain europe"
domain_opt_ifs_g="--domain dynamic --domain dynamic_zoom"
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

# Load conda env for pytrajplot if CONDA_PREFIX not defined
[[ -z $CONDA_PREFIX ]] && conda activate $conda_env
echo CONDA_PREFIX=$CONDA_PREFIX
# Report version
ls -l $CONDA_PREFIX/bin/pytrajplot     # fast
#$CONDA_PREFIX/bin/pytrajplot --version # very slow on balfrin

for basetime in $bt_06 $bt_12 $bt_18 $bt_21 $bt_00 $bt_03 ; do
    yy=${basetime:2:2}
    yymmddhh=${basetime:2}
    hh=${basetime:8:2}

    echo "*****"
    echo Basetime: $(date --utc --date="${basetime:0:8} $hh" "+%F %H UTC")

    # Operational INPUT_DIRs:
    input_dir_icon1=$(echo $store_osm/ICON-CH1-EPS/FCST${yy}/${yymmddhh}_???/lagranto_c/000)
    input_dir_ifs_e=$store_osm/IFS-HRES/IFS-HRES-LAGRANTO${yy}/${yymmddhh}_LIH/lagranto_f
    input_dir_ifs_g=$store_osm/IFS-HRES/IFS-HRES-LAGRANTO${yy}/${yymmddhh}_LIH/lagranto_c

    # Output directories
    output_dir_icon1=$pytrajplot_out/plot_icon1_${basetime}
    output_dir_ifs_e=$pytrajplot_out/plot_ifs_e_${basetime}
    output_dir_ifs_g=$pytrajplot_out/plot_ifs_g_${basetime}

    # Submit jobs
    for model in icon1 ifs_e ifs_g ; do
        [[ $hh == 03 || $hh == 09 || $hh == 15 || $hh == 21 ]] && [[ $model != icon1 ]] && continue
        [[ $hh == 06 || $hh == 18 ]] && [[ $model == ifs_g ]] && continue
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
