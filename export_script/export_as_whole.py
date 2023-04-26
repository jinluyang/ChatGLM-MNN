#!/usr/bin/python

from typing import Mapping, OrderedDict
from transformers.onnx import OnnxConfig
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModel
from transformers.onnx import validate_model_outputs
from torch.onnx import export as torch_export
import torch


model_path = 'YOUR LOCAL PATH'



from pathlib import Path
onnx_path = Path("output2/model.onnx")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
#"LayerNormKernelImpl" not implemented for 'Half'


#random_input = (torch.ones((1,4),dtype=torch.int64), torch.ones((1,2,4),dtype=torch.int64), torch.ones((1,1,4,4),dtype=torch.bool))
#print(random_input[2].shape)
#print(random_input[2][:,1,:])
batch_prompt = ["hello, how are"]
inputs = tokenizer(batch_prompt, return_tensors="pt", padding=True)
random_input = inputs.data

torch_export(model,random_input,"output2/model.onnx",input_names=["input_ids","position_ids","attention_mask"],output_names=['last_hidden_state'],opset_version=15,
             dynamic_axes={"input_ids":{0:"batch",1:"sequence"},
                 "position_ids":{0:"batch",1:"two",2:"sequence2"},
                 "attention_mask":{0:"batch",1:"one",2:"sequence3",3:"sequence"}})  #, verbose=True
