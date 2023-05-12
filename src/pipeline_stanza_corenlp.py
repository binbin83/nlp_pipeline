"""
Stanza CoreNlp Wrapper
Author :  Robin Quillivic
more information available here : https://stanfordnlp.github.io/stanza/corenlp_client.html

"""

from stanza.server import CoreNLPClient
import logging




class StanzaCoreNlpPipeline():
    def __init__(self, stop_word = [], lang ="fr") -> None:
        self.stop_word = stop_word
        self.lang  = lang
        self.client = CoreNLPClient(
            annotators=['tokenize','ssplit','pos','lemma', 'parse'],# 'ner'
            properties=self.lang,endpoint= 'http://localhost:9004',
            timeout=30000,
            memory='6G')
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.test = self.nlp('Ceci est un test')
        self.logger_.info('The pipeline is working !')
    
    def stop_service(self):
        try :
            self.client.stop()
            self.logger_.info('CoreNlp server has stopped')
        except Exception as e: 
            self.logger_.warning(f'Fail to stop CoreNlp server because of {e}')
    
    def compute_annotation(self,text):
        try :
            annotation = self.client.annotate(text)
        except Exception as e :
            self.logger_.warning(f"Fail to annotate text because of {e}")
    
        return annotation
    
    def nlp(self,text):
        text = ' '.join(text.split())# remplacer les doubles espaces
        try :
            extract = Annotation(self.compute_annotation(text))
            extract.extract_annotation(self.stop_word)
        except Exception as e :
            self.logger_.warning(f'Extraction failed because of {e}')
        return {"token":extract.tokens,
                'lemma': extract.lemma,
                'pos': extract.pos,
                'morph': [],
                'xpos': [],
                'doc': extract.annotation}
    
    
class Annotation(object):
    def __init__(self,annotation) -> None:
        self.annotation = annotation
        self.logger_ = logging.getLogger(self.__class__.__name__)
    
    def extract_annotation(self,stop_word = []):
        self.tokens = []
        self.lemma = []
        self.pos = []
        self.morph = []
        #self.simple_pos = []
        try :
            
            for sent in self.annotation.sentence :
                for w in sent.token :
                    if w.word.lower not in stop_word :
                        self.tokens.append(w.word)
                        self.lemma.append(w.lemma)
                        self.pos.append((w.word,w.pos))
                        #self.simple_pos.append(w.pos)

        except Exception as e :
            self.logger_.warning(f'Fail to extract Annotation content because of {e}')
    


