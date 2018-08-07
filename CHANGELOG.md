Thunor Changelog
================

## v0.1.10 - 2018-08-06

### New features

* IncuCyte Zoom format parser
* Colour by dataset on two-dataset plots
* Mann-Whitney U test on bar plots with two groups
* One-way ANOVA on box plots
* Support for 1536 well plates

### Bugfixes

* Various file parser improvements
* Sort box plots ascending order, to match bar plots

## v0.1.9 - 2018-03-08

### Improvements

* Only use control wells from same plate(s) as experimental data wells
* Viability curves now use 3 parameter model, with upper plateau fixed at 1

### Bugfixes

* Fix axis label on viability plot

## v0.1.8 - 2018-03-05

### New features

* Viability calculation and plots (end point or other single time)
* Output to CSV format

### Improvements

* Curve fits now identify the failure mode (whether numerical, QC failure, or statistical rejection)
* Updated documentation and tutorial
* Thunor version saved in HDF5 files to aid reproducibility
* More unit tests

### Bugfixes

* Fix DRC plot on dataset without controls

## v0.1.7 - 2018-02-15

### New features

* Packaged renamed to Thunor

## v0.1.6 - 2018-01-11

### New features

* Box plot of control DIP rates by plate (for QC)
* Plate map layout with DIP rate heat map

### Bugfixes

* Format drug combinations properly in plot title
* Set drug effect E_N to N/A where EC_N is N/A, e.g. for EC100

## v0.1.5 - 2017-10-30

### Improvements

* Drug combination support in HDF5 export
* Original plate well number is now saved in HDF5 files

### Bugfixes

* Fix HDF5 export for datasets with control wells
* dataset cell_lines property now returns control-only cell lines as well as
 cell lines with experimental data

## v0.1.4 - 2017-08-31

### Improvements

* Demonstrate dataset filtering options and examples in tutorial

### Bugfixes

* Fix identification of controls in drug combination CSV uploads
* Fix DIP rate overlay on timecourses when no control data are present

## v0.1.3 - 2017-08-30

### New features

* User defined min/max dose for activity area/area under curve plots
* plotly_to_dataframe function for converting plotly data to pandas (and 
ultimately to CSV)
* Box plots for aggregating parameter values over cell lines and/or drugs
* Drug combination time course plots
* Plots comparing a parameter across two datasets
* Plots with custom IC, EC, E values
* Jupyter notebook tutorial added
* Started unit test suite
* Example data added

### Improvements

* fit_dip_params now returns a pandas dataframe
* EC50s out of measured range are highlighted on parameter plots
* Multiple plot layout and annotation improvements
* Control wells are now defined as any well where the dose for every drug is
 zero
* PyDRC data structures now support drug combinations (although this is not 
fully supported in the analysis functions and plot interface yet)

### Bugfixes

## v0.1.2 - 2017-06-24

### New features

* Activity area as new DIP curve metric

## v0.1.1 - 2017-06-23

### New features

* `read_hdf` can now read from buffers as well as files
* DIP rate fit now returns first timepoint used in fit and y-intercept
* DIP rate overlay option for timecoure plots

## v0.1.0 - 2017-06-20

* Initial version
