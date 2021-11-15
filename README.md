# FogNet
A multiscale 3D CNN with double-branch dense block and attention mechanism for fog prediction

## Publications

[(2021) FogNet: A multiscale 3D CNN with double-branch dense block and attention mechanism for fog prediction](https://www.sciencedirect.com/science/article/pii/S2666827021000190)

Citation:

    @article{kamangir2021fognet,
        title={FogNet: A multiscale 3D CNN with double-branch dense block and attention mechanism for fog prediction},
        author={Kamangir, Hamid and Collins, Waylon and Tissot, Philippe and King, Scott A and Dinh, Hue Thi Hong and Durham, Niall and Rizzo, James},
        journal={Machine Learning with Applications},
        pages={100038},
        year={2021},
        publisher={Elsevier}
    }

## Todo

- [x] Add source code to repo by June 1
- [ ] Add scripts to download, format data

## Optimal threshold

The trained model outputs probabilities for:

- Class 0: fog
- Class 1: non-fog

These always sum to 1. The typical output is a non-fog prediction of 0.99999
Instead of using 0.5 to separate fog and non-fog, an optimal threshold is determined 
by finding the threshold that maximizes the skill score. 

After training, the optimal threshold will be found in the training output directory. 
(`DIRECTORY/test_training_0_report.txt`). You can use this with the script `applyThreshold.py` to convert prediction probabilities to a selected class. 

## Prediction task parameters

Options exist to customize the exact prediction task that the model is for. 

**Time horizon (hours):** how many hours ahead is the prediction? The data supports 6, 12, and 24. 

**Visibility class (meters):** Visibility threshold to be considered fog to predict with model. 
The targets file supports classes 1660, 3200, 6400. 

## Sample trained model

The subdirectory `trained_model` includes the outputs of training FogNet.
The trained model was for 24-hour time horizon, and 1600 meter visibility class. 

## Download & format data

**Data availability:** [North American Mesoscale (NAM) 12km avalable in grib2 format](ftp://ftp.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.YYYYMMDD)

Steps to download and format for FogNet coming soon!

## Installation (Linux)

The following are 2 examples of installation using Anaconda to manage a virtual environment.
The versions of TensorFlow must match the correct CUDA installation.
It is non-trivial to maintain multiple versions of CUDA on a system, without the use of isolated environments.

First, [install Anaconda by following their documentation](https://docs.anaconda.com/anaconda/install/linux/). 

#### For Ubuntu 14.04

    # Create environment
    conda create --name fognet python=3.7.7
    # Activate environment
    conda activate fognet
    # Instal CUDA to use GPU
    conda install -c anaconda cudatoolkit=10.1
    # Install tensorflow
    conda install tensorflow-gpu==2.1.0
    # Install other python packages
    pip install matplotlib seaborn netCDF4 scikit-learn
    # Fix HD5 package compatability, revert to older version
    pip install 'h5py==2.10.0'

#### For Ubuntu 18.08

    # Create environment
    conda create --name fognet python=3.8
    # Activate environment
    conda activate fognet
    # Install CUDA for GPU support
    conda install -c anaconda cudatoolkit=10.1
    # Install tensorflow
    pip install tensorflow-gpu==2.3
    # Install other python packages
    pip install matplotlib seaborn netCDF4 sklearn

## Quickstart

The following scripts have additional options not shown. 
Review the options of each with `--help`. For example, `python src/train.py --help`.

    # If not already, activate environment
    conda activate fognet

**Train model from scratch**

	python src/train.py \           
                    --num_gpus 4 \ # Number of GPUs to use 
                    -o test        # Path to output trained model weights, reports

**Prediction & evaluation (training data) with provided pre-trained model**

    # Prediction
    python src/eval.py \
        -w trained_model/single_gpu_weights.h5  \  # pre-trained weights
        -d ../fognn/Dataset/24HOURS/INPUT/      \  # See data download steps for path
        -l 2018,2019,2020                       \  # training data years  (selects files from data dir)
        -t 24                                   \  # prediction lead time (selects files from data dir)
        -o trained_model/test_preds.csv

    # Inspect result
    head trained_model/test_preds.csv

        pred_fog,pred_non
        1.6408861e-06,0.99999833
        2.7183332e-05,0.9999728
        4.1975437e-07,0.9999995
        7.1297074e-10,1.0
        2.9359464e-06,0.999997
        8.034047e-08,0.9999999
        9.052269e-09,1.0
        8.032033e-10,1.0
        5.1191854e-08,1.0


    # Convert prediction to binary class (apply optimal threshold)
    python src/applyThreshold.py           \  
        -p trained_model/test_preds.csv    \  # Generated from prediction step, above
        -t 0.193                           \  # Found in `trained_model/run_training_0_report.txt`
        -o trained_model/test_classes.csv 

    # Inspect result
    head trained_model/test_classes.csv

        pred_fog,pred_non,pred_class,pred_className
        1.6408861e-06,0.99999833,1,non-fog
        2.7183332e-05,0.9999728,1,non-fog
        4.1975437e-07,0.9999995,1,non-fog
        7.1297074e-10,1.0,1,non-fog
        2.9359464e-06,0.999997,1,non-fog
        8.034047e-08,0.9999999,1,non-fog
        9.052269e-09,1.0,1,non-fog
        8.032033e-10,1.0,1,non-fog
        5.1191854e-08,1.0,1,non-fog


    # Calculate metrics & add column with outcome (hit, false alarm, miss, correct-reject)
    python src/calcMetrics.py              \
        -p trained_model/test_classes.csv  \  # Generated from threshold step, above
        -t trained_model/test_targets.txt  \  # List of target classes, one per line
        -o trained_model/test_outcomes.csv

    # Inspect result
    head trained_model/test_outcomes.csv

        pred_fog,pred_non,pred_class,pred_className,target_class,target_className,outcome,outcome_name
        1.6408861e-06,0.99999833,1,non-fog,1,non-fog,d,correct-reject
        2.7183332e-05,0.9999728,1,non-fog,1,non-fog,d,correct-reject
        4.1975437e-07,0.9999995,1,non-fog,1,non-fog,d,correct-reject
        7.1297074e-10,1.0,1,non-fog,1,non-fog,d,correct-reject
        2.9359464e-06,0.999997,1,non-fog,1,non-fog,d,correct-reject
        8.034047e-08,0.9999999,1,non-fog,1,non-fog,d,correct-reject
        9.052269e-09,1.0,1,non-fog,1,non-fog,d,correct-reject
        8.032033e-10,1.0,1,non-fog,1,non-fog,d,correct-reject
        5.1191854e-08,1.0,1,non-fog,1,non-fog,d,correct-reject


## (Experimental!!) Run XAI methods

**Run XAI script: Channel-wise PartitionShap**
    
[Click here for information on Channel-wise PartitionShap and how to visualize the output](https://github.com/conrad-blucher-institute/partitionshap-multiband-demo).

    # Install packages
    pip install git+https://github.com/conrad-blucher-institute/shap.git

    # We had to downgrade a package for compatiability
    pip uninstall h5py
    pip install h5py==2.10.0
    
    # PartitionShap too slow to run all 2228 instances
    # Already made a file with 4 selected
    # head trained_model/shap_sample_instances.txt

        100
        200
        300

    # Run Channel-wise PartitionShap on those instances (from 2019 data)
    python src/xaiPartitionShap.py               \
        -d ../fognn/Dataset/24HOURS/INPUT/       \        # See data download steps for path
        --cases trained_model/shap_sample_instances.txt \ # List of instances (row numbers) to explain
        -w trained_model/single_gpu_weights.h5   \        # Pretrained weights
        -l 2019                                  \        # 2019 test data
        -t 24                                    \        # Prediction lead time
        --max_evaluations 10000                    \        # Number of SHAP evaluations. More -> smaller superpixels, but longer time.
        --masker color=0.5                       \        # Simulate removing input features by replacement with value 0.5
        -o trained_model/shap_sample_values.pickle


## Data format

Example dataset format, where $TIME=24 and $LABEL=2009

	Dataset/
	├── 24HOURS
	│   ├── INPUT
	│   │   ├── NETCDF_MIXED_CUBE_2019_24.npz
	│   │   ├── NETCDF_NAM_CUBE_2019_PhG1_24.npz
	│   │   ├── NETCDF_NAM_CUBE_2019_PhG2_24.npz
	│   │   ├── NETCDF_NAM_CUBE_2019_PhG3_24.npz
	│   │   ├── NETCDF_NAM_CUBE_2019_PhG4_24.npz
	│   │   └── NETCDF_SST_CUBE_2019.npz
	│   ├── NAMES
	│   │   ├── MurfileNames2019_24.txt
	│   │   ├── NamFileNames2019_24.txt
	│   └── TARGET
	│       └── target2019_24.csv
	├── HREF
	│   └── href-vis-2019.csv
	├── NAMES
	│   ├── MurfileNames2019.txt
	│   ├── NamFileNames2019.txt
	│   └── NamFileNames2019.txt
	├── STAT
	│   └── Stat_2019.txt
	└── TARGET
	    ├── target2019.csv
	    ├── Target.csv
	    ├── Testing.csv
	    ├── Training.csv
	    └── Validation.csv

