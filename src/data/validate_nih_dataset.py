import numpy as np
import pandas as pd
import uuid


class ValidateNIHData:
    """
    Data validation means checking the accuracy and quality of source data before training a new model version. 
    It ensures that anomalies that are infrequent or manifested in incremental data are not silently ignored..
    Inpsect the dataset and purge anamolies
    """

    def __init__(self):

        self.nih_columns = nih_column_names = {
            'Image Index': 'image_name',
            'Finding Labels': 'finding_label',
            'Follow-up #': 'follow_up_num',
            'Patient ID': 'patient_id',
            'Patient Age': 'age',
            'Patient Gender': 'gender',
            'View Position': 'view_position',
            'OriginalImage[Width': 'image_width',
            'Height]': 'image_height',
            'OriginalImagePixelSpacing[x': 'x_spacing',
            'y]': 'y_spacing'
        }

    def read_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        self.nih_xrays_df = pd.read_csv(raw_data_path)

    def process_data(self, stable=True):
        """Process raw data into useful files for model.
        Cleans the ground dataset
        """
        self.nih_xrays_df = self.nih_xrays_df.rename(columns=self.nih_columns)
        self.nih_xrays_df.drop(columns=['Unnamed: 11'], inplace=True)

        # fix data errors - remove patient record with age greater than 100
        self.nih_xrays_df = self.nih_xrays_df[self.nih_xrays_df['age'] <= 100]

    def write_data(self, processed_data_path):
        """Write processed data to directory."""

        self.nih_xrays_df.to_csv(processed_data_path)
