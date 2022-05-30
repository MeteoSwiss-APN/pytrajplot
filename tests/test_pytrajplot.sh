#!/bin/bash
# Run set of test cases

# Path to input and output directories
input_dir_tests=tests
output_dir_tests=out

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~HRES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# test variable number of altitude pltos
pytrajplot $input_dir_tests/test_hres/4_altitudes $output_dir_tests/test_hres/4_altitudes --datatype png --domain europe
pytrajplot $input_dir_tests/test_hres/3_altitudes $output_dir_tests/test_hres/3_altitudes --datatype png --domain europe
pytrajplot $input_dir_tests/test_hres/2_altitudes $output_dir_tests/test_hres/2_altitudes --datatype png --domain europe
pytrajplot $input_dir_tests/test_hres/1_altitudes $output_dir_tests/test_hres/1_altitudes --datatype png --domain europe

# test HRES backward trajectory
pytrajplot $input_dir_tests/test_hres/backward $output_dir_tests/test_hres/backward --datatype png --domain europe

# test all domains combined w/ HRES model
pytrajplot $input_dir_tests/test_hres/4_altitudes $output_dir_tests/test_hres/4_altitudes --datatype png --domain ch --domain alps --domain centraleurope --domain europe --domain dynamic

# test HRES dateline crossing
pytrajplot $input_dir_tests/test_hres/dateline $output_dir_tests/test_hres/dateline --datatype png --domain dynamic

# test HRES w/o side trajectories and the german case
pytrajplot $input_dir_tests/test_hres/dateline $output_dir_tests/test_hres/dateline/german --datatype png --domain europe --language de

# test HRES Europe with trajectory crossing of zero longitude from east and then leaving domain
# (failed in v1.0.0 due to NaNs in the argument of the numpy max function - v1.0.1 uses nanmax instead)
pytrajplot $input_dir_tests/test_hres/zero_lon_from_east $output_dir_tests/test_hres/zero_lon_from_east

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~COSMO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# test COSMO forward/backward trajectories w/ all COSMO domains (ch, ch_hd, alps)
pytrajplot $input_dir_tests/test_cosmo/forward  $output_dir_tests/test_cosmo/forward --datatype png --domain ch --domain alps
pytrajplot $input_dir_tests/test_cosmo/backward $output_dir_tests/test_cosmo/backward --datatype png --domain ch --domain alps
