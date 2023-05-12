
import pandas as pd
import os

def aggregate_nlp_data(codes:list,data_folder = "",select =['text','token','lemma','pos'],target_col='part_1_1',speaker = None):
    """_summary_

    Args:
        codes (_type_): _description_
        data_folder (str, optional): _description_. Defaults to 'Z://data//nlp_files'.
        select (list, optional): _description_. Defaults to ['text','token','lemma','pos'].
        part (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    col = ['code','speaker']+select
    df_nlp = pd.DataFrame(columns=col)
    i=0 
   
    for code in codes:
        try:
            
            df = pd.read_pickle(os.path.join(data_folder,code+'_'+target_col+'.pkl'))
          
            #filter speaker if wanted
            if speaker == 'enqueteur':
                result = build_nlp_dict_from_df(df,speaker=speaker)
            else :
                result = build_nlp_dict_from_df(df,speaker=code)
            result['code'] = code
            result = { key: result[key] for key in col }
            df_nlp.loc[i] = pd.Series(result)
            i+=1
        except Exception as e :
         print(f'{code} non traité à cause de: {e}')

    return df_nlp


def build_nlp_dict_from_df(df:pd.DataFrame,speaker:str) ->dict:

    df_speaker = df[df['speaker']==speaker]

    text = '\n'.join(df_speaker['text']).replace('*','')
    token = df_speaker['token'].sum()
    lemma = df_speaker['lemma'].sum()
    pos = df_speaker['pos'].sum()
    morph = df_speaker['morph'].sum()
    try :
       doc = df_speaker['doc'].tolist()
    except :
        doc = []
    result = {
      'text': text,
      'token': token,
      'lemma':lemma,
      'pos':pos,
      "morph" : morph,
      "doc":doc
    }
    result['speaker']  = speaker
    
    return result

def get_file_id(folder:str)->list:
    """Return the list of uniquie id in a folder

    Args:
        folder (str): _description_

    Returns:
        list: _description_
    """
    files = os.listdir(folder)
    ids = list(set([file.split('_')[0] for file in files]))
    return ids



def load_corpus(config:dict)->list:
    """Load corpus from a config file

    Args:
        config (dict): _description_

    Returns:
        list: _description_
    """
    data_folder = os.path.join(config['outputs']['outputs_folder'],
                                     config['outputs']['nlp_folder'],
                                     config['pipeline']['method'])
    codes = get_file_id(data_folder)

    data = aggregate_nlp_data(codes, select = [config["embeddings"]['unit']],
                              data_folder = data_folder,
                              target_col=config['pipeline']['col'])
    
    
    corpus = data[config["embeddings"]['unit']].tolist()
    
    
    return corpus
