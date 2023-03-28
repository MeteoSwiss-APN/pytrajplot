#!/bin/bash
# Run set of test cases

# Name of this script
tag=$(basename $0)

# Path to input and output directories
input_dir_tests=tests
output_dir_tests=local/$(basename $CONDA_PREFIX)

if [[ -z $CONDA_PREFIX ]] ; then
    echo $tag: "Activate conda environment before running this script."
    exit 1
fi

# Print info
echo $tag: "Running pytrajplot with test input"
echo "Input from: $input_dir_tests"
echo "Output to:  $output_dir_tests"
which pytrajplot
pytrajplot --version

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~HRES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
echo "Test variable number of altitude plots"
pytrajplot $input_dir_tests/test_hres/4_altitudes $output_dir_tests/test_hres/4_altitudes --datatype png --domain europe
pytrajplot $input_dir_tests/test_hres/3_altitudes $output_dir_tests/test_hres/3_altitudes --datatype png --domain europe
pytrajplot $input_dir_tests/test_hres/2_altitudes $output_dir_tests/test_hres/2_altitudes --datatype png --domain europe
pytrajplot $input_dir_tests/test_hres/1_altitudes $output_dir_tests/test_hres/1_altitudes --datatype png --domain europe

echo "Test HRES backward trajectory"
pytrajplot $input_dir_tests/test_hres/backward $output_dir_tests/test_hres/backward --datatype png --domain europe

echo "Test all domains combined w/ HRES model"
pytrajplot $input_dir_tests/test_hres/4_altitudes $output_dir_tests/test_hres/4_altitudes --datatype png --domain ch --domain alps --domain centraleurope --domain europe --domain dynamic

echo "Test HRES dateline crossing"
pytrajplot $input_dir_tests/test_hres/dateline $output_dir_tests/test_hres/dateline --datatype png --domain dynamic

echo "Test HRES w/o side trajectories and the german case"
pytrajplot $input_dir_tests/test_hres/dateline $output_dir_tests/test_hres/dateline/german --datatype png --domain europe --language de

echo "Test HRES Europe with trajectory crossing of zero longitude from east and then leaving domain"
# (failed in v1.0.0 due to NaNs in the argument of the numpy max function - v1.0.1 uses nanmax instead)
pytrajplot $input_dir_tests/test_hres/zero_lon_from_east $output_dir_tests/test_hres/zero_lon_from_east

echo "Test HRES trajectory ending at zero longitude and next ancillary trajectory starting with abs(lon) > 20 deg"
# (failed in v1.1.1 due to erroneously detecting a date line crossing but corresponding dictionary reamining undefined)
pytrajplot $input_dir_tests/test_hres/zero_last_lon $output_dir_tests/test_hres/zero_last_lon

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~COSMO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
echo "Test COSMO forward/backward trajectories w/ all COSMO domains (ch, ch_hd, alps)"
pytrajplot $input_dir_tests/test_cosmo/forward  $output_dir_tests/test_cosmo/forward --datatype png --domain ch --domain alps
pytrajplot $input_dir_tests/test_cosmo/backward $output_dir_tests/test_cosmo/backward --datatype png --domain ch --domain alps
