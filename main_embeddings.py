"""
script to build embedding

Author :  Quillivic Robin
Date:  30/03/2020

"""
import os

import time
import shutil


from src.utils import load_config_file, load_logger
from src.pipeline_embeddings import BuildEmbeddings
from src.utils import flatten


from src.aggregate import load_corpus



if __name__ == "__main__":
    t_start = time.time()
    config = load_config_file("config.yaml")
    logger = load_logger("logging_config.yaml")

    builder = BuildEmbeddings(config=config)
    saving_folder  =  os.path.join(config['outputs']["outputs_folder"], config['outputs']["models_folder"])

    model_name  =  config['embeddings']["model_name"]
    model_type = config['embeddings']["model_type"]
    corpus_name = config['pipeline']['method'] +"_"+ config['embeddings']['unit'] +"_"+ config['pipeline']['col']
    
    logger.info(f"""Config and builder loaded  ! Parameters are :
     - model : {model_type},
     - model_name : {model_name},
     - corpus : {corpus_name} """)

    # load data
    corpus = load_corpus(config)
    logger.info(f'corpus loaded, {len(corpus)} documennts, {len(flatten(corpus))} words ')
    
    builder.train_and_save(corpus = corpus, method=model_type, model_name = model_name)

   
    t_end = time.time()
    d = round(t_end-t_start,2)/60
    try :
        saving_folder = os.path.join(builder.saving_folder,builder.method,model_name)
        print(saving_folder)
        shutil.move('nlp_pipeline.log',os.path.join(saving_folder,'pipeline.log'))
        logger.info('Process terminated, logfile saved in folder')
    except Exception as e:
        logger.warning(f'Process terminated, logfile could not be  saved in folder{saving_folder} bacause of {e}')
    logger.info(f'Process takes {d} minutes')