#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~HRES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# test variable number of altitude pltos
pytrajplot zmichel/test_hres/4_altitudes/ zmichel/test_hres/plots --datatype png --domain europe
pytrajplot zmichel/test_hres/3_altitudes/ zmichel/test_hres/plots --datatype png --domain europe
pytrajplot zmichel/test_hres/2_altitudes/ zmichel/test_hres/plots --datatype png --domain europe
pytrajplot zmichel/test_hres/1_altitudes/ zmichel/test_hres/plots --datatype png --domain europe

# test HRES backward trajectory
pytrajplot zmichel/test_hres/backward/ zmichel/test_hres/plots --datatype png --domain europe

# test all domains combined w/ HRES model
pytrajplot zmichel/test_hres/4_altitudes/ zmichel/test_hres/plots --datatype png --domain ch --domain ch_hd --domain alps --domain centraleurope --domain europe --domain dynamic

# test HRES dateline crossing
pytrajplot zmichel/test_hres/dateline/ zmichel/test_hres/plots --datatype png --domain dynamic

# test HRES w/o side trajectories and the german case
pytrajplot zmichel/test_hres/dateline/ zmichel/test_hres/plots --datatype png --domain europe --language de

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~COSMO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# test COSMO forward/backward trajectories w/ all COSMO domains (ch, ch_hd, alps)
pytrajplot zmichel/test_cosmo/forward  zmichel/test_cosmo/plots --datatype png --domain ch --domain ch_hd --domain alps
pytrajplot zmichel/test_cosmo/backward zmichel/test_cosmo/plots --datatype png --domain ch --domain ch_hd --domain alps
