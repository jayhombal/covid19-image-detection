# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from validate_nih_dataset import ValidateNIHData
from prepare_nih_dataset import PrepareNIHData


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    prepare_nih = PrepareNIHData(MIN_CASES_FLAG=False)
    logger.info('PrepareNIH - reading validatedd dataset')
    prepare_nih.read_data(input_filepath)
    logger.info('PrepareNIH - Prepare data')
    prepare_nih.process_data()
    logger.info('PrepareNIH - saving validated data')
    prepare_nih.write_data(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
