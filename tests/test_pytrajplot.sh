#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~HRES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# test variable number of altitude pltos
pytrajplot tests/test_hres/4_altitudes/ tests/test_hres/plots --datatype png --domain europe
pytrajplot tests/test_hres/3_altitudes/ tests/test_hres/plots --datatype png --domain europe
pytrajplot tests/test_hres/2_altitudes/ tests/test_hres/plots --datatype png --domain europe
pytrajplot tests/test_hres/1_altitudes/ tests/test_hres/plots --datatype png --domain europe

# test HRES backward trajectory
pytrajplot tests/test_hres/backward/ tests/test_hres/plots --datatype png --domain europe

# test all domains combined w/ HRES model
pytrajplot tests/test_hres/4_altitudes/ tests/test_hres/plots --datatype png --domain ch --domain ch_hd --domain alps --domain centraleurope --domain europe --domain dynamic

# test HRES dateline crossing
pytrajplot tests/test_hres/dateline/ tests/test_hres/plots --datatype png --domain dynamic

# test HRES w/o side trajectories and the german case
pytrajplot tests/test_hres/dateline/ tests/test_hres/plots --datatype png --domain europe --language de

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~COSMO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# test COSMO forward/backward trajectories w/ all COSMO domains (ch, ch_hd, alps)
pytrajplot tests/test_cosmo/forward  tests/test_cosmo/plots --datatype png --domain ch --domain ch_hd --domain alps
pytrajplot tests/test_cosmo/backward tests/test_cosmo/plots --datatype png --domain ch --domain ch_hd --domain alps
