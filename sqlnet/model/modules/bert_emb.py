from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
#import logging
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

#logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
#                    datefmt = '%m/%d/%Y %H:%M:%S',
#                    level = logging.INFO)
#logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, text_a, text_b):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.text_a = text_a
        self.text_b = text_b


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        #tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = [char for char in example.text_a]
        tokens_b = None
        if example.text_b:
            #tokens_b = tokenizer.tokenize(example.text_b)
            tokens_b = [char for char in example.text_b]

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        '''
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
        '''
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                text_a = example.text_a,
                text_b = example.text_b
                ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(str_list):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    for i,s in enumerate(str_list):
        text_a = "".join(s)
        text_b = None
        examples.append(InputExample(unique_id=i, text_a=text_a, text_b=text_b))
    """
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            data=json.loads(line)
            #line = line.strip()
            if file_type=="q":
                text_a = data["question"]
                text_b = None
            #m = re.match(r"^(.*) \|\|\| (.*)$", line)
            #if m is None:
                #text_a = line
            #else:
            #    text_a = m.group(1)
            #    text_b = m.group(2)
                for cond in data["sql"]["conds"]:
                    if not text_b:
                        text_b=cond[-1]
                    else:
                        text_b= text_b+","+cond[-1]
                examples.append(
                    InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
                unique_id += 1
            else:
                for header in data["header"]:
                    text_a = header
                    text_b = None
                    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
                    unique_id += 1
    """
    return examples

def read_q_examples(inputfile):
    
    examples = []
    with open(inputfile, "r", encoding='utf-8') as reader:
        unique_id = 0
        max_len = 0
        while True:
            line = reader.readline()
            if not line:
                break
            data=json.loads(line)
            text_a = data["question"]
            text_b = None
            #m = re.match(r"^(.*) \|\|\| (.*)$", line)
            #if m is None:
                #text_a = line
            #else:
            #    text_a = m.group(1)
            #    text_b = m.group(2)
            text_b = data["title"]
            max_len = max_len if max_len >= len(text_a) + len(text_b) else len(text_a) + len(text_b)
            examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples, max_len+3

def read_col_examples(inputfile):
    examples = []
    with open(inputfile, "r", encoding='utf-8') as reader:
        unique_id = 0
        max_len=0
        while True:
            line = reader.readline()
            if not line:
                break
            data=json.loads(line)
            for char in data["header"]:
                text_a = char
                text_b = data["title"]
                max_len = max_len if max_len >= len(text_a) + len(text_b) else len(text_a) + len(text_b)
            #m = re.match(r"^(.*) \|\|\| (.*)$", line)
            #if m is None:
                #text_a = line
            #else:
            #    text_a = m.group(1)
            #    text_b = m.group(2)
                examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
                unique_id += 1
    return examples, max_len+3

def gen_q_emb(inputfile, outputfile,
    do_lower_case = True, batch_size = 16, gpu = False):

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=do_lower_case)

    examples, max_seq_len = read_q_examples(inputfile)

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_len, tokenizer=tokenizer)

    model = BertModel.from_pretrained('bert-base-chinese')
    model.to(device)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    
    eval_sampler = SequentialSampler(eval_data)
    
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    output_json={}

    with open(outputfile, "w") as writer:
        for input_ids, input_mask, example_indices in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                
                #unique_id = int(feature.unique_id)
            # feature = unique_id_to_feature[unique_id]
            
            #output_json["linex_index"] = unique_id
                all_out_features = []
                for (i, token) in enumerate(feature.tokens):
                    if i == 0 or feature.tokens[i-1] != "[SEP]":
                #all_layers = []
                #for (j, layer_index) in enumerate(layer_indexes):
                #token = feature.tokens[1]
                        layer_output = all_encoder_layers[int(12)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                #layers = collections.OrderedDict()
                #layers["index"] = layer_index
                #layers["values"] = [
                #    round(x.item(), 6) for x in layer_output[i]
                #]
                #all_layers.append(layers)
                #out_features = collections.OrderedDict()
                #out_features["token"] = token
                #out_features["layers"] = all_layers
                        out_features=[
                            round(x.item(), 6) for x in layer_output[i]
                        ]
                        all_out_features.append(out_features)
                    else:
                        break   
                output_json[feature.text_a] = all_out_features
        writer.write(json.dumps(output_json))
    print("bert_question_emb is generated in file : %s"%outputfile)
    return 0

def gen_col_emb(inputfile, outputfile,
    do_lower_case = True, batch_size = 16, gpu = False):

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=do_lower_case)
    
    
    examples, max_seq_len = read_col_examples(inputfile)

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_len, tokenizer=tokenizer)
    
    model = BertModel.from_pretrained('bert-base-chinese')
    model.to(device)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    
    eval_sampler = SequentialSampler(eval_data)
    
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    output_json={}

    with open(outputfile, "w") as writer:
        for input_ids, input_mask, example_indices in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                
                #unique_id = int(feature.unique_id)
            # feature = unique_id_to_feature[unique_id]
            
            #output_json["linex_index"] = unique_id
                all_out_features = []
                for (i, token) in enumerate(feature.tokens):
                    if i == 0 or feature.tokens[i-1] != "[SEP]":
                #all_layers = []
                #for (j, layer_index) in enumerate(layer_indexes):
                #token = feature.tokens[1]
                        layer_output = all_encoder_layers[int(12)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                #layers = collections.OrderedDict()
                #layers["index"] = layer_index
                #layers["values"] = [
                #    round(x.item(), 6) for x in layer_output[i]
                #]
                #all_layers.append(layers)
                #out_features = collections.OrderedDict()
                #out_features["token"] = token
                #out_features["layers"] = all_layers
                        out_features=[
                            round(x.item(), 6) for x in layer_output[i]
                        ]
                        all_out_features.append(out_features)
                    else:
                        break 
                if feature.text_b not in output_json.keys():
                    output_json[feature.text_b] = {}
                output_json[feature.text_b][feature.text_a]] = all_out_features
        writer.write(json.dumps(output_json))
    print("bert_col_emb is generated in file : %s"%outputfile)
    return 0
    

def get_emb(str_list,max_seq_len,output_file,
    do_lower_case=True,layer_indexes=[1,2,3,4],batch_size=16,gpu=False):


    #if not gpu:
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    n_gpu = torch.cuda.device_count()
    #else:
        #device = torch.device("cuda", 0)
        #n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl')
    #logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))


    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=do_lower_case)

    examples = read_examples(str_list)

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_len, tokenizer=tokenizer)
    #out_questions={}
    #for feat in features:
        #out_questions[feat.unique_id]=feat.tokens[1:-1]

    #unique_id_to_feature = {}
    #for feature in features:
    #    unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained('bert-base-chinese')
    model.to(device)

    #if gpu:
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0],
        #                                                  output_device=0)
    #elif n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    #if not gpu:
    eval_sampler = SequentialSampler(eval_data)
    #else:
        #eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    output_json={}
    with open(output_file, "w") as writer:
        for input_ids, input_mask, example_indices in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                
                #unique_id = int(feature.unique_id)
            # feature = unique_id_to_feature[unique_id]
            
            #output_json["linex_index"] = unique_id
            #all_out_features = []
            #for (i, token) in enumerate(feature.tokens):
                #all_layers = []
                #for (j, layer_index) in enumerate(layer_indexes):
                token = feature.tokens[1]
                layer_output = all_encoder_layers[int(layer_indexes[-1])].detach().cpu().numpy()
                layer_output = layer_output[b]
                #layers = collections.OrderedDict()
                #layers["index"] = layer_index
                #layers["values"] = [
                #    round(x.item(), 6) for x in layer_output[i]
                #]
                #all_layers.append(layers)
                #out_features = collections.OrderedDict()
                #out_features["token"] = token
                #out_features["layers"] = all_layers
                out_features=[
                    round(x.item(), 6) for x in layer_output[1]
                ]
                #all_out_features.append(out_features)
                output_json[token] = out_features
        writer.write(json.dumps(output_json))
    print("bert_emb is generated in file : %s"%output_file)
    return 0
