import yaml
import os
import logging
import logging.config
import pandas as pd

import torch
from src import pipeline_spacy, pipeline_stanza, pipeline_stanza_corenlp, pipeline_transformers


def load_config_file(config_path:str=None) ->dict:
    """load the project config file

    Args:
        config_path (str, optional): _description_. Defaults to None.

    Returns:
        dict: config file
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def load_logger(config_path:str=None) -> logging.Logger:
    """load a logger from a config file

    Args:
        config_path (str, optional): _description_. Defaults to None.

    Returns:
        logging.Logger: _description_
    """

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'logging_config.yaml')
    
    with open(config_path, 'r') as stream :
        log_config = yaml.load(stream, Loader=yaml.FullLoader)

    logging.config.dictConfig(log_config)
    logger = logging.getLogger('nlp_pipeline')

    return logger

def flatten(l):
    """flattent a list of list

    Args:
        l (list): _description_

    Returns:
        list: _description_
    """
    return [item for sublist in l for item in sublist]


def compute_gpu_free_memory()->float:
    """
    compute the free memory of the gpu

    Returns:
        float: _description_
    """
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    
    return  r-a


def load_pipeline(method:str, config:dict, logger:logging.Logger):
    """load a pipeline from a config file"""

    if method in ["spacy", "spacy_trf"]:
        pipeline = pipeline_spacy.SpacyNlpPipeline(**config["pipeline"][method])
    elif method == "stanza" :
        pipeline = pipeline_stanza.StanzaNlpPipeline(**config["pipeline"][method])
    elif method == "stanzaCore":
        pipeline = pipeline_stanza_corenlp.StanzaCoreNlpPipeline(**config["pipeline"][method])
    elif method == "transformer":
        pipeline = pipeline_transformers.TransformersNlpPipeline(**config["pipeline"][method])
    else :
        pipeline = None
        logger.warning('No pipeline were loadind, check the method name')
    return pipeline


def load_data(config:dict)->pd.DataFrame:
    """
    Load data into dataframe using different format

    Args:
        config (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """

    data_path = os.path.join(config['data']['source_folder'],config['data']['source_filename'])
    
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.hdf'):
        data = pd.read_hdf(data_path,key='df')
    elif data_path.endswith('.h5'):
        data = pd.read_hdf(data_path,key='df')
    elif data_path.endswith('.pkl'):
        data = pd.read_pickle(data_path)
    elif data_path.endswith('.xlsx'):
        data = pd.read_excel(data_path)
    elif data_path.endswith('.json'):
        data = pd.read_json(data_path)
    else :
        raise ValueError(f'Format of {data_path} is not supported yet')
    
    return data



