from sqlnet.utils import *
from sqlnet.model.modules.bert_emb import*


def add_title(data = 'train'):
    tablefile = 'data/' + data + '/' + data +'.table.json'
    datafile = 'data/' + data + '/' + data +'.json'
    savafile = 'data/' + data + '/' + data +'.addtitle.json'
    dic={}
    with open(tablefile, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            dic[data['id']] = data['title']
    with open(datafile, 'r', encoding='utf-8') as f1, open(savafile, 'w') as f2:
        while True:
            output ={}
            line = f1.readline()
            if not line:
                break
            data = json.loads(line)
            output['title'] = dic[data['table_id']]
            output['question'] = data['question']
            f2.write(json.dumps(output)+'/n')
    return savafile

def concate(qfile, colfile, file,outfile):
    f1 = open(colfile)
    col = json.load(f1)
    f1.close()
    f2 = open(qfile)
    q = json.load(f2)
    f2.close()
    dic={}
    with open(file,'r', encoding='utf-8') as cf, open(outfile,'w') as out:
        while True:
            line = cf.readline()
            if not line:
                break
  
            data = json.loads(line)
            dic[data['question']] = {'question':q[data['question']],'col':col[data['title']]}
        out.write(json.dumps(dic))
                 

if __name__ == '__main__':

    #for train
    trainfile = add_title('train')
    ret_q = gen_q_emb(trainfile, 'data/train/train_bert_q.json')
    ret_col = gen_col_emb('data/train/train.tables.json', 'data/train/train_bert_col.json')
    concate('data/train/train_bert_q.json','data/train/train_bert_col.json', trainfile, 'data/train/new_train_bert.json')

    #for test
    testfile = add_title('test')
    ret_q = gen_q_emb(testfile, 'data/test/test_bert_q.json')
    ret_col = gen_col_emb('data/test/test.tables.json', 'data/test/test_bert_col.json')
    concate('data/test/test_bert_q.json','data/test/test_bert_col.json', testfile, 'data/test/new_test_bert.json')
    #for val
    valfile = add_title('val')
    ret_q = gen_q_emb(valfile, 'data/val/val_bert_q.json')
    ret_col = gen_col_emb('data/val/val.tables.json', 'data/val/val_bert_col.json')
    concate('data/val/val_bert_q.json','data/val/val_bert_col.json', valfile, 'data/val/new_val_bert.json')

    f1 = open('data/train/new_train_bert.json')
    train = json.load(f1)
    f1.close()
    f2 = open('data/val/new_val_bert.json')
    val = json.load(f2)
    f2.close()
    dictmerg=train.copy()
    dictmerg.update(val)
    with open('data/new_bert_char_embedding.json','w') as out:
        out.write(json.dumps(dictmerg))
    
    

