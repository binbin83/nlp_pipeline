"""

Author :  Robin Quillivic
more information available here : 
tags :
    - gilf : https://huggingface.co/gilf/french-camembert-postag-model
    - 
lemmes :
    - leff :  https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer 

more models available here : 
https://huggingface.co/models?language=fr&pipeline_tag=token-classification&sort=downloads

"""

# Exemple
"""

"""



import transformers
import logging

from transformers import AutoTokenizer, AutoModelForTokenClassification
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

from transformers import pipeline



class TransformersNlpPipeline():
    def __init__(self,model_name = 'gilf/french-camembert-postag-model', stop_word = ['*'], use_gpu=False) -> None:
        self.use_gpu = use_gpu
        self.stop_word  = stop_word
        self.model_name = model_name
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.load_model()
      
        try :
            self.test = self.nlp('Ceci est un test')
            self.logger_.info('The pipeline is working !')
        except  Exception as e :
            self.logger_.warning('Test failed, the pipeline is not working')
    
    def load_model(self):
        try : 
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.pipeline =  pipeline('ner', model=self.model, tokenizer= self.tokenizer, grouped_entities=True)
            self.lemmatizer = FrenchLefffLemmatizer()
        except Exception as e:
            self.logger_.warning(f'Fail to load model because of {e}')
            raise e
    
    def convert_to_UPOS(self,tag):
        if tag in ['ADJ','ADJWH'] :
            return 'ADJ'
        elif tag in ['ADV','ADVWH'] :
            return 'ADJ'
        elif tag in ['CC']:
            return 'CCONJ'
        elif tag in ['CLO','CLR','CLS','PRO','PROREL','PROWH']:
            return 'PRON'
        elif tag in ['CS']:
            return 'SCONJ'
        elif tag in ['DET','DETWH']:
            return 'DET'
        elif tag in ['ET','U']:
            return 'X'
        elif tag in ['I']:
            return 'INTJ'
        elif tag in ['PONCT']:
            return 'PUNCT'
        elif tag in ['V','VIMP','VPP','VPR','VS']:
            return 'VERB'
        elif tag in ['NC']:
            return 'NOUN'
        elif tag in ['NPP']:
            return 'PROPN'
        elif tag in ['P','P+D']:
            return 'ADP'
        else :
            return 'X'
    
    def convert_to_morph(self,tag):
        if tag in ['VPR']:
            return 'Tense=Pres'
        elif tag in ['VPP'] :
            return 'Tense=Past'
        elif tag in ['VIMP'] :
            return 'Mood=Imp'
        elif tag in ['VINF'] :
            return 'Mood=Inf'
        elif tag in  ['VS'] :
            return 'Mood=Subj'
        else :
            return False
    
    def simple_tag(self,tag):
        if tag in ['NC'] :
            return 'n'
        elif tag in ['ADV','ADVWH'] :
            return 'r'
        elif tag in ['V','VIMP','VPP','VPR','VS']:
            return 'v'
        elif tag in ['ADJ','ADJWH']:
            return 'a'
        else :
            return False
        
    def nlp(self,text):
        try :
            tokens = []
            lemma = []
            morphs = []
            pos = []
            xpos = []
            doc = self.pipeline(text)
            for text in doc :
                word = text['word']
                tag = text['entity_group']
                morph = self.convert_to_morph(tag)
                upos = self.convert_to_UPOS(tag)
                lem_test = self.lemmatizer.lemmatize(word,self.simple_tag(tag))
                if len(lem_test)>0:
                    lemme = lem_test
                else :
                    lemme = self.lemmatizer.lemmatize(word)
                if word.lower() not in self.stop_word :
                    xpos.append(tag)
                    tokens.append(word)
                    pos.append((word,upos ))
                    morphs.append((word, upos,morph))
                    lemma.append(lemme)
        except Exception as e :
            self.logger_.warning(f'Extraction failed because of {e}')

        return {"token": tokens,
                'lemma': lemma,
                'pos': pos,
                'xpos':xpos,
                'morph':morphs,
                'doc': None}