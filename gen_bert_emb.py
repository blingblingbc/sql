from sqlnet.utils import *
from sqlnet.model.modules.bert_emb import*



if __name__ == '__main__':

    words = transform_word_emb('data/char_embedding.json')
    ret = get_emb(words,3,'data/bert_char_embedding.json',gpu=True)
    
