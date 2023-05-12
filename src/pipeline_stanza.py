"""
Stanza Wrapper
Author :  Robin Quillivic
more information available here : https://stanfordnlp.github.io/stanza/pipeline.html

"""

import stanza
import logging




class StanzaNlpPipeline():
    def __init__(self,lang = "fr",stop_word = ['*'], use_gpu=False) -> None:
        self.use_gpu = use_gpu
        self.stop_word  = stop_word
        self.lang = lang
        self.pipeline = stanza.Pipeline(lang=self.lang, processors='tokenize,mwt,pos,lemma',use_gpu = self.use_gpu)
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.test = self.pipeline('Ceci est un test')
        self.logger_.info('The pipeline is working !')
    
    
    def nlp(self,text):
        try :
            text = ' '.join(text.split())# remplacer les doubles espaces
            tokens = []
            lemma = []
            morphs = []
            pos = []
            xpos = []
            doc = self.pipeline(text)
            for sent in doc.sentences :
                for word in sent.words :
                    if word.text.lower() not in self.stop_word :
                        tokens.append(word.text)
                        lemma.append(word.lemma)
                        pos.append((word.text,word.upos))
                        xpos.append((word.text,word.xpos))
                        if word.feats :
                            morphs.append((word.text,word.upos,word.feats))
                        else :
                            morphs.append(False)
        except Exception as e :
            self.logger_.warning(f'Extraction failed because of {e}')

        return {"token": tokens,
                'lemma': lemma,
                'pos': pos,
                'xpos':xpos,
                'morph':morphs,
                'doc':doc}
    
    

    


