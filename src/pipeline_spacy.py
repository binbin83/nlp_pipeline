"""
Spacy Wrapper
Author :  Robin Quillivic
more information available here : https://spacy.io/

"""

# Exemple
"""
from pipeline_spacy import SpacyNlpPipeline

text = " je suis un exemple de text très simple"
pipeline = SpacyNlpPipeline(model_name='fr_dep_news_trf', use_gpu = True)
result = pipeline.nlp(text)
result
"""



import spacy
import logging
from thinc.api import set_gpu_allocator, require_gpu




class SpacyNlpPipeline():
    def __init__(self,model_name = 'fr_core_news_lg', stop_word = ['*'], use_gpu=False) -> None:
        self.use_gpu = use_gpu
        self.stop_word  = stop_word
        self.model_name = model_name
        self.logger_ = logging.getLogger(self.__class__.__name__)

        if self.model_name == 'fr_dep_news_trf' and self.use_gpu == True :
            set_gpu_allocator("pytorch")
            require_gpu()
            self.logger_.info('The Gpu is used because transformer was chosen!')
        
        self.pipeline = spacy.load(self.model_name)
        try :
            self.test = self.pipeline('Ceci est un test')
            self.logger_.info('The pipeline is working !')
        except  Exception as e :
            self.logger_.warning('Test failed, the pipeline is not working')
    
    
    def nlp(self,text):
        try :
            text = ' '.join(text.split())# remplacer les doubles espaces
            tokens = []
            lemma = []
            morphs = []
            pos = []
            xpos = []
            doc = self.pipeline(text)
            for word in doc :
                if word.text.lower() not in self.stop_word :
                    tokens.append(word.text)
                    lemma.append(word.lemma_)
                    pos.append((word.text, word.tag_))
                    if word.has_morph() :
                            morphs.append((word.text, word.tag_, str(word.morph)))
                    else :
                        morphs.append[False]
        except Exception as e :
            self.logger_.warning(f'Extraction failed because of {e}')

        result = {"token": tokens,
                'lemma': lemma,
                'pos': pos,
                'xpos':xpos,
                'morph':morphs,
                'doc':doc}
        
        if self.model_name == 'fr_dep_news_trf' and self.use_gpu == True :
            doc._.trf_data = None


        return result
    
    

    


