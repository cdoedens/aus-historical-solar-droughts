from compute_solar import 

def test_tilting_panel():
    """
    Test the module which computes solar PV
    """

    # Give the model some known input values

    ghi= np.array([60.23, 60.51, 60.81, 61.06, 61.34, 61.58, 
                          61.83, 62.12, 62.36,62.69])

    dhi = np.array([30.66, 30.79, 30.86, 30.93, 31.05, 31.12, 
                          31.19, 31.25, 31.33, 31.44])
    
    dni = np.array([282.75, 283.4, 284.7, 285.57, 286.35, 287.1, 
                          287.95, 289.32, 290.05, 291.19])

    time_1d = np.repeat([dt.datetime(2022,1,31,20,40)],len(dni))

    lat_1d = np.repeat([-17.38],len(dni))

    lon_1d = np.array([42.74, 142.76, 142.78, 142.8 , 142.82, 142.84,
                       142.86, 142.88, 142.9 , 142.92])

    # Specify the target output values
    
    target = {'mean': 2.367783,
              'max' : 2.461892,
              'min' : 2.279921}

    LOG.info('Testing tilting_panel_pr')

    actual_ideal_ratio_t = tilting_panel_pr(
        pv_model = 'Canadian_Solar_CS5P_220M___2009_',
        inverter_model = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        time=time_1d,
        lat=lat_1d,
        lon=lon_1d
    )  

    # Check the outputs match the corresponding known values
    # Drop the first 'nan' values from the output
    actual_ideal_ratio_t = actual_ideal_ratio_t[~np.isnan(actual_ideal_ratio_t)]

    assert np.isclose(actual_ideal_ratio_t.mean(), target['mean'], 1e-6)
    assert np.isclose(actual_ideal_ratio_t.max(), target['max'], 1e-6)
    assert np.isclose(actual_ideal_ratio_t.min(), target['min'], 1e-6)