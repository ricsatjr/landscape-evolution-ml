# create environment using miniconda, install dependencies, and activate environment

```
conda env create -f environment.yml
conda activate leml
```
leml is the name of the environment as written in the environment.yml file


# generating landscapes

`python pipeline/01_generate_landscapes.py --job-id <job-id> --n-landscapes 10 --output-dir data/landscapes/`

A single execution of the above command generates 10,000 candidate parameter sets, applies the constraints, and takes the first 10 valid ones to generate 10 time series of elevation grids, with each time series represented by 20 snapshots of the evolving landcape. The last grid in each timeseries (*-99.npy) are the ones used in the succeeding analyses. 

<job-id> provides the random seed for reproducibility. The list of job-id's used to produce the steady state landscapes for this study are found in landscape-batch-ids.csv

LE parameters are saved as pickle files; elevation grids are saved as numpy files.

# generating features

## stage 1

`python pipeline/02_extract_features.py --stage rasnet --job-id <job-id> --data-dir data/landscapes/ --rasnet-dir data/rasnet/`

A single execution of the above command generates the processed raster and network files corresponding to the landscapes generated using <job-id>. The raster and network files will be used in stage 2 to extract the features from the landscape. 

Outputs are saved and exported as pickle files.

## stage 2
`python pipeline/02_extract_features.py --stage features --job-id all --data-dir data/rasnet/ --output-dir data/features/`

A single execution of the above command generates a dataframe of features and labels from all files in the rasnet directory. 

Dataframe is saved and exported as pickle files. 

# ML model training

`python pipeline/03_train_models.py --data-dir data/features/ --output-dir data/models --labels u_ks kh_ks`
`
The above command trains the machine learning models using the features pickle file in data/features using u/ks and kh/ks as target labels

`python pipeline/03_train_models.py --data-dir data/features/ --output-dir data/models --labels u ks kh`

The command trains the machine learning models using u, ks, and kh as target labels

Both commands produce a dictionary of model outputs, saved and exported as a pickle file. 

# Journal figures


## ML model performance
```
python pipeline/plot_ml_model_performance.py   --pkl <path-to-nested-cv-results-for-ratio-labels>   --pkl <path-to-nested-cv-results-for-individual-labels>  --pred-model mlp --data-dir data/features --row-xlim 0.90 1.00 --row-xlim -0.10 0.20   --row-max-xticks 3 --row-max-xticks 3   --axis-labels --row-height-ratios 1.35 1.35 1.75 1.75   --row-gap 0.40 --top 0.955 --bottom 0.055   --width-cm 19 --height-cm 22.5   --out figures/model_performance.jpg
```
The above commands generates a plot showing the performance of trained machine learning models. It requires the paths to the pickle files containing the results of model training, as well as the folder containing the features pickle file. 

This 

