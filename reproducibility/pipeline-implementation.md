# creating environment

Code block creates the miniconda environment and installs dependencies from repository file, activates the environment, and sets the working path to the repository's root. 

```
conda env create -f <path to repository>/environment.yml
conda activate leml
cd <path to repository>
```


# generating landscapes

Code generates 10,000 candidate parameter sets, applies the constraints, and takes the first 10 valid ones to generate 10 time series of elevation grids, with each time series represented by 20 snapshots of the evolving landcape. The last grid in each timeseries (*-99.npy) are the ones used in the succeeding analyses. 

<job-id> provides the random seed for reproducibility. The list of all job-id's used to produce the steady state landscapes for this study are found in `reproducibility/landscape-batch-ids.csv`. 

LE parameters are saved as pickle files; elevation grids are saved as numpy files.

`python pipeline/01_generate_landscapes.py --job-id <job-id> --n-landscapes 10 --output-dir data/landscapes/`

# generating features

## stage 1

Code generates the processed raster and network files corresponding to the landscapes generated using <job-id>. The raster and network files will be used in stage 2 to extract the features from the landscape. Random noise (in the form of elevation measurement error) of magnitude defined by --elev-error (in meters) is added to the landscape elevation grid prior to processing.

Outputs are saved and exported as pickle files.

For elevation error = 10m:
`python pipeline/02_extract_features.py --stage rasnet --job-id <job-id> --data-dir data/landscapes/ --rasnet-dir data/rasnet/n10 --elev-error 10`

For elevation error = 1m:
`python pipeline/02_extract_features.py --stage rasnet --job-id <job-id> --data-dir data/landscapes/ --rasnet-dir data/rasnet/n01 --elev-error 1`



## stage 2


Generates a dataframe of features and labels from all files in the rasnet directory. 

Dataframe is saved and exported as pickle files. 


For rasnet files with --elev-error = 10:
`python pipeline/02_extract_features.py --stage features --job-id all --data-dir data/rasnet/n10 --output-dir data/features/n10`

For rasnet files with --elev-error = 1: 
`python pipeline/02_extract_features.py --stage features --job-id all --data-dir data/rasnet/n01 --output-dir data/features/n01`



# ML model training

Code trains the machine learning models using the features pickle file in data/features using u/ks and kh/ks as target labels. Use `--labels u ks kh` to train models using individual parameters as target labels. 

A dictionary of model outputs is saved and exported as a pickle file. 


`python pipeline/03_train_models.py --data-dir data/features/n10 --output-dir data/models/n10/ratio --labels u_ks kh_ks --test-fraction 0.2`

`python pipeline/03_train_models.py --data-dir data/features/n10 --output-dir data/models/n10/indiv --labels u_ks kh_ks --test-fraction 0.2`


`python pipeline/03_train_models.py --data-dir data/features/n01 --output-dir data/models/n01/ratio --labels u_ks kh_ks --test-fraction 0.2`
`
# Feature comparison

Compare features derived from n10 rasnets with those from n01 rasnets; specifically how the distribution of each feature in n01 varies in relation to the distribution of the corresponding feature in n10.  

`python pipeline/03b_compare_features.py --n10-features-dir data/features/n10 --n01-features-dir data/features/n01 --out data/features/paired-feature-comp.csv --models-pkl data/models/n10/ratio/nested-cv-results-full-u_ks-kh_ks-9b33cab.pkl --subset train`


# feature importance

`python pipeline/04_feature_importance.py --mode explore --features-dir data/features/n10/ --models-pkl data/models/n10/train-1200/nested-cv-results-full-u_ks-kh_ks-9b33cab.pkl --output-dir data/models/n10/train-1200/reduced --cluster-threshold 0.25 --cluster-selection random`

`python pipeline/04_feature_importance.py --mode reduced --features-dir data/features/n10/ --models-pkl data/models/n10/train-1200/nested-cv-results-full-u_ks-kh_ks-9b33cab.pkl --reduced-models-pkl data/models/n10/train-1200/reduced/nested-cv-results-reduced-u_ks-kh_ks-9b33cab.pkl --output-dir data/models/n10/train-1200/reduced --cluster-threshold 0.25 --cluster-selection domain --domain-features n0 crv_kurt Rb Rb0 hyp_int Rl0 Z_cv Z_skew htcrv_min crv_mean Rl Z_mean htcrv_max`

          

Rb hyp_int l0_mean Rl Z_mean htcrv_max n0 Rb0 Z_cv Z_skew crv_kurt htcrv_min crv_mean grd0_mean







# Journal figures


## Parameter space and representative landscapes


`python pipeline/plot_param_landscapes.py   --data-dir data/features   --rasnet <path-to-rasnet-data>   --rasnet <path-to-rasnet-data>   --rasnet <path-to-rasnet-data>   --rasnet <path-to-rasnet-data>  --out figures/param-space.jpg --summary`
 
This code plots the constrained parameter space, and let's you select four representative landscapes to illustrate how LEM parameters influence the topography. 

Here, we used the following rasnet files: rasnet-n10-122023-5-99.pkl,rasnet-n10-122013-0-99.pkl, rasnet-n10-121720-5-99.pkl, and rasnet-n10-123117-3-99.pkl 


## ML model performance
```
python pipeline/plot_ml_model_performance.py   --pkl <path-to-nested-cv-results-for-ratio-labels>   --pkl <path-to-nested-cv-results-for-individual-labels>  --pred-model mlp --data-dir data/features --row-xlim 0.90 1.00 --row-xlim -0.10 0.20   --row-max-xticks 3 --row-max-xticks 3   --axis-labels --row-height-ratios 1.35 1.35 1.75 1.75   --row-gap 0.40 --top 0.955 --bottom 0.055   --width-cm 19 --height-cm 22.5   --out figures/model_performance.jpg
```
The above command generates a plot showing the performance of trained machine learning models. It requires the paths to the pickle files containing the results of model training, as well as the folder containing the features pickle file. 

## Training set size

`python pipeline/plot_trainN_performance.py --model-root data/models/n10 --pattern 'nested-cv-results-full-u_ks-kh_ks-*.pkl' --output-dir figures --show-final`

The above command generates a plot showing the performance of algorithms as a function of the size of the training set, N. For N in {300,600,900,1200}, we computed and plotted the generalized performance of each algorithm. The left and right subplots represent the performances on the U/Ks and Kh/Ks labels. 






## Network generation

`python pipeline/plot_network_extraction.py  --rasnet <path-to-rasnet-file> --target-order 5 --order-lw-step 0.35     --legend-loc under-b --legend-shrink 0.75     --output-dir figures  --basemap logA --basemap-alpha 0.5 --basemap-cbar-shrink 0.6`


The above command generates a plot with subplos showing a simple cell-to-cell flowpath network, and its correspondng reach-based network after removal of through-nodes. The bottom subplots show an area of one of the synthetic landscapes (rasnet-n10-122023-5-99.pkl) overlain with the extracted networks similar to subplots A and B. 

 

