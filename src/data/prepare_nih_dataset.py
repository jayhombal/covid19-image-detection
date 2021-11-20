import numpy as np
import pandas as pd
from  glob import glob
import os
import logging

from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

class PrepareNIHData:
    """
    Data preparation is the process of cleaning and transforming raw data prior to 
    processing and analysis. It is an important step prior to processing and often 
    involves reformatting data, making corrections to data and the combining of data
    sets to enrich data.
    """


    def __init__(self):
        self.logger = logging.getLogger("PrepareNIHData")
        self.logger.info('making final data set from raw data')  
        self.MIN_CASES_FLAG = True
        self.MIN_CLASSES = 500
                
        self.nih_required_columns = nih_column_names =  {
            'image_name',
            'finding_label',
            'patient_id',
            'path'
        }

    def read_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        self.nih_xrays_df = pd.read_csv(raw_data_path)


    def process_data(self, stable=True):
        """
        Prepare dataset
        """
        #add image path column to the dataset 
        # create a set of all image paths
        image_path = 'data/raw'
        all_image_paths = {os.path.basename(x): x for x in glob( os.path.join(image_path, 'images*', '*', '*.png'))}

        self.nih_xrays_df['path'] = self.nih_xrays_df['image_name'].map(all_image_paths.get)
        
        print('count of raw images paths and rows in NIH dataset :', \
            str(len(all_image_paths)), ', Total Headers', \
            str(self.nih_xrays_df.shape[0]))

        # replace the 'No Finding' with '' value = why?
        self.nih_xrays_df['finding_label'] = self.nih_xrays_df['finding_label'].map(lambda x: x.replace('No Finding', 'NoFinding'))
        
        # Get fourteen unique diagnosis
        # It is a function that takes a series of iterables and returns one iterable
        # The asterisk "*" is used in Python to define a variable number of arguments. 
        # The asterisk character has to precede a variable identifier in the parameter list 
        from itertools import chain
        all_labels = np.unique(list(chain(*self.nih_xrays_df['finding_label'].map(lambda x: x.split('|')).tolist())))
        
        # remove the empty label
        all_labels = [x for x in all_labels if len(x)>0]
        self.logger.info('All Labels ({}): {}'.format(len(all_labels), all_labels))
        
        for label in all_labels:
             self.nih_xrays_df[label]= self.nih_xrays_df['finding_label'].map(lambda finding: 1 if label in finding else 0)
            
        # Apply the min_cases logic

        if self.MIN_CASES_FLAG:
            all_labels_with_min_cases = [label for label in all_labels \
                                     if self.nih_xrays_df[label].sum() > self.MIN_CLASSES]
            self.logger.info(f'finding labels with min cases: {len(all_labels_with_min_cases)}')  
            self.logger.info([(label, int(self.nih_xrays_df[label].sum())) for label in all_labels_with_min_cases])
            self.nih_xrays_df= self.nih_xrays_df[self.nih_xrays_df['finding_label'].isin(all_labels_with_min_cases)]

        
        self.nih_xrays_df = self.nih_xrays_df[self.nih_required_columns]
        group_shuffle_split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

        for train_idx, valid_idx in group_shuffle_split.split(self.nih_xrays_df[:None],\
            groups=self.nih_xrays_df[:None]['patient_id'].values):
            self.train_df = self.nih_xrays_df.iloc[train_idx]
            self.valid_df = self.nih_xrays_df.iloc[valid_idx]
    
    def write_data(self, processed_data_path):
        """Write processed data to directory."""
        self.logger.info("Write processed data to data/processed directory")
        self.nih_xrays_df.to_csv(processed_data_path, header= True, index=False)
        self.train_df.to_csv('data/processed/prepared_train_data_entry_2017.csv' , header= True, index=False) 
        self.valid_df.to_csv('data/processed/prepared_valid_data_entry_2017.csv' , header= True, index=False) 