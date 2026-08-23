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


