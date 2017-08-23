# Dose response curves and drug induced proliferation (DIP) rates in Python

## Implementation

PyDRC makes extensive use of [pandas](http://pandas.pydata.org/) and 
[plotly](http://plot.ly/python/), so familiarity with those libraries is 
recommended.

## Requirements

 * **numpy**
 * **scipy**
 * **pandas**
 * **plotly**
 * **pytables** for HDF5 format read/write
 * **seaborn** for colour palette selection

## Examples

### Load data

From CSV (Vanderbilt HTS core format)
 
    from pydrc.io import read_vanderbilt_hts
    hts_data = read_vanderbilt_hts('mydataset.csv')

From HDF5 file

    from pydrc.io import read_hdf
    hts_data = read_hdf(‘mydataset.h5’)
    
### Save data

To HDF5 file

    from pydrc.io import write_hdf
    write_hdf(hts_data, ‘mydataset.h5’)
    
### Calculate DIP rates and dose response parameters

    from pydrc.dip import dip_rates, dip_fit_params
    
    ctrl_dip_data, expt_dip_data = dip_rates(hts_data)
    fit_params = dip_fit_params(ctrl_dip_data, expt_dip_data)
    
### Plot results with plotly

Each of the `plot_X` functions returns a plotly `Figure` object which can be
 visualised in a number of ways. Here, we use the offline `plot` function 
 which saves an HTML file and opens it in a web browser. It's also possible 
 to view plots in a Jupyter notebook. See the 
 [plotly documentation](https://plot.ly/python/offline/) for more 
 information on the latter approach. 

    from pydrc.plots import plot_dip, plot_dip_params, plot_time_course
    from plotly.offline import plot
    
DIP rate dose response

    plot(plot_dip(fit_params))
    
DIP rate dose response curve parameters (IC50, EC50, AUC etc.)

    plot(plot_dip_params(fit_params, 'auc'))
    
Time course plot for the 'Cell count' assay (this is a bit more work at the 
moment as we need to manually filter a drug/cell line/assay combination)

    df_doses_filtered = hts_data['doses'].xs(['abemaciclib', 'BT20'],
                                             level=['drug', 'cell_line'],
                                             drop_level=False)
    df_controls_filtered = hts_data['controls'].loc['Cell count', 'BT20']
    df_assays_filtered = hts_data['assays'].loc['Cell count']
    plot(plot_time_course(df_doses_filtered, df_assays_filtered, \ 
                          df_controls_filtered))
