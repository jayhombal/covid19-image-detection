.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = covid19-image-detection
PYTHON_INTERPRETER = python
NIH_XRAYS_KAGGLE_URL = https://www.kaggle.com/nih-chest-xrays/data/download

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

#make Download dataset
download_nih: 
	@echo ">>> Downloading NIH X-ray data from Kaggle..."
	kaggle datasets download --force -d nih-chest-xrays/data
	mv data.zip data/external

get_nih_images: download_nih
	@echo ">>> Unzipping NIH X-ray data."
	mkdir data/raw/nih
	unzip data/external/data.zip -d data/raw/nih

## Download covid images
download_covid:
	@echo ">>> Downloading Covid19 X-ray data from Kaggle..."
	kaggle datasets download --force -d pranavraikokte/covid19-image-dataset
	kaggle datasets download -d mushaxyz/covid19-customized-xray-dataset
	mkdir -p data/external
	mv covid19-image-dataset.zip data/external/
	mv covid19-customized-xray-dataset.zip data/external
	@echo ">>> Unzipping Covid-19 X-ray image data."
	rm -rf data/raw/Covid19-dataset
	unzip data/external/covid19-image-dataset.zip -d data/raw/
	unzip data/external/covid19-customized-xray-dataset.zip -d data/raw/
	

get_covid19_images: 
	unzip data/external/covid19-customized-xray-dataset.zip -d data/raw/
	mv -f data/raw/COVID19\ CUSTOMIZED\ X-RAY\ DATASET/  data/raw/covid19-images
	# mv data/raw/Covid19-dataset/train/Covid/*.* data/raw/covid19-images/COVID19/
	# mv data/raw/Covid19-dataset/train/Normal/*.* data/raw/covid19-images/NORMAL/
	# mv data/raw/Covid19-dataset/train/Viral\ Pneumonia/*.* data/raw/covid19-images/PNEUMONIA/
	# mv data/raw/Covid19-dataset/test/Covid/*.* data/raw/covid19-images/COVID19/
	# mv data/raw/Covid19-dataset/test/Normal/*.* data/raw/covid19-images/NORMAL/
	# mv data/raw/Covid19-dataset/test/Viral\ Pneumonia/*.* data/raw/covid19-images/PNEUMONIA/
	mv -f data/raw/covid19-images/PNEUMONIA/ data/raw/covid19-images/Pneumonia/
	mv -f data/raw/covid19-images/NORMAL/ data/raw/covid19-images/Normal/
	mv -f data/raw/covid19-images/COVID19/ data/raw/covid19-images/Covid19/
	# rm -rf data/raw/Covid19-dataset


## Validate Dataset
validate_nih_images: 
	$(PYTHON_INTERPRETER) src/data/validate_dataset.py data/raw/nih/Data_Entry_2017.csv data/interim/interim_data_entry_2017.csv

prepare_nih_images: 
	$(PYTHON_INTERPRETER) src/data/prepare_dataset.py data/interim/interim_data_entry_2017.csv data/processed/prepared_data_entry_2017.csv




## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py
	mkdir -p data
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p data/external
	mkdir -p data/interim
	mkdir -p logs
	mkdir -p logs/fit

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
