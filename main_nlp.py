"""
Author : Quillivic Robin
this file enables to apply nlp pipeline such as stanza, spacy, standford or transformer 
to a list of interaction and strore it into jsonfiles

Date : 30/03/2022
"""
import os
import shutil
import pandas as pd

import time

from src.utils import compute_gpu_free_memory, load_config_file, load_logger, load_pipeline, load_data







if __name__=="__main__":
    t_start = time.time()

    # config and logger
    logger = load_logger("logging_config.yaml")
    config = load_config_file("config.yaml")


    logger.info('loading config file....')
    
   
    method = config['pipeline']['method']
    target_col = config['pipeline']['col']
    data_folder = config["data"]['source_folder']

    outputs_nlp_folder = os.path.join(config['outputs']["outputs_folder"],config['outputs']["nlp_folder"])
    spe_nlp_folder = os.path.join(outputs_nlp_folder,method)

    
    logger.info(f"""The config file is loaded, the method use will be: {method}, 
    the data folder is :{data_folder}, the saving folder will be: {spe_nlp_folder} and,
    the selected column is {target_col}""")

    # create folder
    if not os.path.exists(outputs_nlp_folder):
        os.mkdir(outputs_nlp_folder)
        logger.info('{outputs_nlp_folder} was created ! ')
    if not os.path.exists(spe_nlp_folder):
        os.mkdir(spe_nlp_folder)
        logger.info('{spe_nlp_folder} was created ! ')
    
    # data
    logger.info('Loading the data...')
    data = load_data(config)
    logger.info(f'Data loaded, ie {len(data)} lines')

    # nlp_pipeline
    pipeline = load_pipeline(method, config, logger=logger)
    logger.info('The pipeline is now loaded !')
    
    for i in range(len(data)) :
        line = data.iloc[i]
        code = line['code']
        interaction_list = line[target_col]
        if not os.path.exists(os.path.join(spe_nlp_folder,code+'_'+target_col+'.json')) :
            for interaction in interaction_list :
                result = pipeline.nlp(interaction['text'])
                interaction.update(result)
            logger.info(f"{code} analysis is terminated")
            # saving 
            df = pd.DataFrame(interaction_list)
            file_name = os.path.join(spe_nlp_folder,code+'_'+target_col+'.pkl')
            df.to_pickle(file_name)
            logger.info(f'{file_name} Saved')
            logger.info(f'Free memory on GPU is: {compute_gpu_free_memory()}')

    t_end = time.time()
    d = round(t_end-t_start,2)/60
    try :
        shutil.move('./nlp_pipeline.log',os.path.join(spe_nlp_folder,'nlp_pipeline.log'))
        logger.info('Process terminated, logfile saved in folder')
    except Exception as e:
        logger.warning('Process terminated, logfile could not be  saved in folder')
    logger.info(f'Process takes {d} minutes')