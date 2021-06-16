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
- [ ] Make data available
- [ ] Make number of GPUs variable
- [ ] Port to TensorFlow 2
- [ ] Improve documentation
- [ ] Add XAI scripts

# Notes

- Currently hard-coded to train with 4 GPUs
- Some absolute paths are used in this documentation that will be cleaned up before external release.
- Right now, members of the FogNet team can find the data on TAMUCC's deep learning at `/data1/fog/Dataset`

## Optimal threshold

The trained model outputs probabilities for:

- Class 0: fog
- Class 1: non-fog

These always sum to 1. The typical output is a non-fog prediction of 0.99999
Instead of using 0.5 to separate fog and non-fog, an optimal threshold is determined 
by finding the threshold that maximizes the skill score. 

After training, the optimal threshold will be found in the training output directory. 
(`DIRECTORY/run_testing_0_report.txt`). You can use this with the script `applyThreshold.py` to convert prediction probabilities to a selected class. 

## Data (coming soon)

The dataset is quite large, and has not yet been made available. 
We plan to release (1) the data used for publication results and (2) the scripts used to pull and format the data NWP data.

## Prediction task parameters

Options exist to customize the exact prediction task that the model is for. 

**Time horizon (hours):** how many hours ahead is the prediction? The data supports 6, 12, and 24. 

**Visibility class (meters):** Visibility threshold to be considered fog to predict with model. 
The targets file supports classes 1660, 3200, 6400. 

## Sample trained model

The subdirectory `trained_model` includes the outputs of training FogNet.
The trained model was for 24-hour time horizon, and 1600 meter visibility class. 

## Quickstart

	# Activate python environment
	export PATH=/home/hkamangir/anaconda3/bin:$PATH
	source activate deep-learning

	# Training
	#   (--force is used to overwrite existing test output directory)
	python src/train.py -o test --force

    # Convert multi-GPU model (for training) to single-GPU model (for prediction)
    python src/multi2single.py -w  test/weights.h5              \ # Saved weights of trained FogNet
                    -d /data1/fog/fognn/Dataset/24HOURS/INPUT/  \ # Path to FogNet data
                    -l 2019               \                       # The "$LABEL", files have format 'NETCDF_NAM_CUBE_$LABEL_PhG3_$TIME.npz'
                    -t 24                 \                       # The "$TIME",  files have format 'NETCDF_NAM_CUBE_$LABEL_PhG3_$TIME.npz'
                   -o trained_weights.h5

    # Prediction
    python src/eval.py -w trained_weights.h5                   \ # Saved weights of trained FogNet
                   -d /data1/fog/fognn/Dataset/24HOURS/INPUT/  \ # Path to FogNet data
                   -l 2019               \                       # The "$LABEL", files have format 'NETCDF_NAM_CUBE_$LABEL_PhG3_$TIME.npz'
                   -t 24                 \                       # The "$TIME",  files have format 'NETCDF_NAM_CUBE_$LABEL_PhG3_$TIME.npz'
                   -o output_preds.txt   \                       # Path to save the output predictions
                   --filters 24          \                       # Must match the trained weights' model
                   --dropout 0.3                                 # Must match the trained weights' model

    # Binary prediction (apply optimal threshold)
    python src/applyThreshold.py \
        -p output_preds.txt \         # Generated from prediction step, above
        -t 0.129            \         # Found in `test/run_testing_0_report.txt`
        -o output_class.txt           # Output with added column for selected class

    # Calculate performance metrics & determine optimal prediction threshold
    # (In case you deleted the training report, or want to recalculate metrics)
    COMING SOON

    # Generate custom data cube with selected instances
    COMING SOON


## Training script `train.py` options

	Usage: train.py [options]
	
	Options:
	  -h, --help            show this help message and exit
	  -n NAME, --name=NAME  Model name [default = test].
	  -d DIRECTORY, --directory=DIRECTORY
	                        Fog dataset directory [default = /data1/fog/Dataset/].
	  -o OUTPUT_DIRECTORY, --output_directory=OUTPUT_DIRECTORY
	                        Output results directory [default = none].
	  --force               Force overwrite of existing output directory [default
	                        = False].
	  -t TIME_HORIZON, --time_horizon=TIME_HORIZON
	                        Prediction time horizon [default = 24].
	  --train_years=TRAIN_YEARS
	                        Comma-delimited list of training years [default =
	                        2013,2014,2015,2016,2017].
	  --val_years=VAL_YEARS
	                        Comma-delimited list of validation years [default =
	                        2009,2010,2011,2012].
	  --test_years=TEST_YEARS
	                        Comma-delimited list of testing years [default =
	                        2018,2019,2020].
	  -v VISIBILITY_CLASS, --visibility_class=VISIBILITY_CLASS
	                        Visibility class [default = 0].
	  -b BATCH_SIZE, --batch_size=BATCH_SIZE
	                        Training batch size [default = 32].
	  -e EPOCHS, --epochs=EPOCHS
	                        Training epochs [default = 30].
	  --learning_rate=LEARNING_RATE
	                        Learning rate [default = 0.0009].
	  --weight_penalty=WEIGHT_PENALTY
	                        Weight penalty [default = 0.001].
	  --filters=FILTERS     Number of filters [default = 24].
	  --dropout=DROPOUT     Droput rate [default = 0.3].


## Prediction script `eval.py` options

    Usage: eval.py [options]

    Options: 
      -h, --help            show this help message and exit 
      -w WEIGHTS, --weights=WEIGHTS
                            Path to trained model weights
      -d DIRECTORY, --directory=DIRECTORY
                            Path to directory with fog data cubes
      -l LABEL, --label=LABEL
                            Unique label to identify data cubes in data directory
      -t TIME_HORIZON, --time_horizon=TIME_HORIZON
                            Prediction time horizon
      -o OUTPUT_PREDICTIONS, --output_predictions=OUTPUT_PREDICTIONS
                            Path to file to save predictions
      --filters=FILTERS     Number of filters [default = 24].
      --dropout=DROPOUT     Droput rate [default = 0.3].
      -v, --verbose         Verbose output 


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

