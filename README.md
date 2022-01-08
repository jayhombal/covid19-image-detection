covid19-image-detection-tfl
==============================

Image recognition with Convolutional Neural Networks (CNN) is also an extremely valuable skill worthy of acquisition and practice. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


STEPS to RUN THE PROJECT:

Pre-requisites:

1. Install make
   Make is pre-installed in Ubuntu and MacOS, windows users refer https://community.chocolatey.org/packages/make 
2. Kaggle API client setup 
   https://github.com/Kaggle/kaggle-api

3. Setup Python environment

    Ubuntu:
        Install conda and setup a python environment with tensorflow python3.9.7
    Mac OS M1:
        Follow instructurions
            - https://towardsdatascience.com/installing-tensorflow-on-the-m1-mac-410bb36b776
            - https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-mac-metal-jul-2021.ipynb

4. create data folder with subdirectories raw,interim,processed,external the project folder

Project Code Orginzation:
Cookiecutter is a command-line utility that creates projects from cookiecutters (project templates), e.g. Python package projects, LaTeX documents, etc.

Installed and created the project template using Cookiecutter:
Follow instructions from https://ericbassett.tech/cookiecutter-data-science-crash-course/

Validating and pre-processing NIH X-ray metadata dataset
Following instructions use make tool, run commands from from your terminal from your project folder

Setup
Setup python environment

1. Validate Python is installed and create required directories  
    Run: make test_environment  

Data Extraction:(execute only once)

2. Download and unzip the NIH X-ray images in data/raw    
    Run: make get_nih_images   

Data Validation:(execute only once)

3. Validate Dataset (rename columns and delete patient record with age greater than 100)   
    Run: make validate_nih_images   

Data Prepartion:(execute only once)

4. Prepare Dataset (add path attribute, split dataset into train and validation dataset)
    Run: make prepare_nih_images

This proudces the three output files in processed folder:

1. prepared_data_entry_2017.csv (full dataset)
2. prepared_train_data_entry_2017.csv (train_dataset)
3. prepared_valid_data_entry_2017.csv (validation_dataset)

Next, we use prepared_train_data_entry_2017.csv and prepared_valid_data_entry_2017.csv files to retrain CNN model architectures pre-trained using IMAGENET database
