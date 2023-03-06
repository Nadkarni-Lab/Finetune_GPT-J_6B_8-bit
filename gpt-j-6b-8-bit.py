# -*- coding: utf-8 -*-
"""
finetune-gpt-j-6B-8bit.ipynb
https://colab.research.google.com/drive/1ft6wQU0BhqG5PRlwgaZJv2VukKKjU4Es
### Fine-tuning 6-Billion GPT-J in colab with LoRA and 8-bit compression
(https://huggingface.co/EleutherAI/gpt-j-6B) with limited memory. A 
https://huggingface.co/hivemind/gpt-j-6B-8bit)
This notebook is a proof of concept for fine-tuning 
[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B) with limited memory. 
A detailed explanation of how it works can be found in [this model card]
(https://huggingface.co/hivemind/gpt-j-6B-8bit).
"""
print("starting import libs")
from loguru import logger
import transformers
import torch
import torch.nn.functional as F
from torch import nn
print("done with torch")
from torch.cuda.amp import custom_fwd, custom_bwd
print("halfway through import libs. importing bitsandbytes")
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
print("done with bitsandbytes")
from tqdm.auto import tqdm
from datasets import load_dataset
from bitsandbytes.optim import Adam8bit
import time, os

##form gustavocortal https://colab.research.google.com/drive/1ymTfojEPhibqsXl-Ik8V0SZb8AXewIlV#scrollTo=bX35XMqY1JfY
import pandas as pd
from sklearn.model_selection import train_test_split


print("done with import libs")
# ---------------------> Converting the model to 8 bits <------------------- #
"""
We convert EleutherAI's GPT-J-6B model to 8 bits using facebook's [bitsandbytes](https://github.com/facebookresearch/bitsandbytes) library. 
This reduces the model's size from 20Gb down to just 6Gb.
Note that we don't convert linear layer biases to 8 bit as they take up less that 1% of the model's weight anyway.
"""

class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias

    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias) 
        print("input is:")
        print(input)	

        print("output is:")
        print(output)


        print("In Linear what about self.adapter??")

        if self.adapter:
            print("self.adapter is TRUE")
            print(self.adapter(input))
            print("appending to output")
            output2 = self.adapter(input)
            print("now output2 is:")
            print(output2)
            output= torch.cat((output, output2), 0)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"



class DequantizeAndLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None

    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"

def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)


def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)



logger.info("Initializing Transformers model GPTJB")
transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

# ---------------------> Loading EleutherAI/gpt-j-6B config and tokenizer <------------------- #
## use pre-saved model from /sc/arion/projects/EHR_ML/GPTJ-6B/gpt-j-6B/
#config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
#tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
logger.info("reading config and tokenizer from saved")
save_dir_config = "/sc/arion/projects/EHR_ML/GPTJ-6B/saved_gpt-j-6B"
#config.save_pretrained(save_dir_config)

save_dir_tokenizer = "/sc/arion/projects/EHR_ML/GPTJ-6B/saved_gpt-j-6B"
#tokenizer.save_pretrained(save_dir_tokenizer)


config = transformers.GPTJConfig.from_pretrained("/sc/arion/projects/EHR_ML/GPTJ-6B/gpt-j-6B")
logger.info("Config: {}".format(config))

tokenizer = transformers.AutoTokenizer.from_pretrained("/sc/arion/projects/EHR_ML/GPTJ-6B/gpt-j-6B")
#tokenizer = transformers.AutoTokenizer.from_pretrained("/sc/arion/projects/EHR_ML/GPTJ-6B/gpt-j-6B", pad_token='<|PAD|>')
logger.info("Tokenizer: {}".format(tokenizer))

config.pad_token_id = config.eos_token_id
tokenizer.pad_token = config.pad_token_id

# ---------------------> Downloading gpt-j-6B-8bit model from huggingface <------------------- #
#gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

# ----------------> Saving gpt-j-6B-8bit model to server <-----------------#
#save_dir = "/home/paperspace/project/saved_models_gpt-j-6B-8bit/gpt-j-6B"
#gpt.save_pretrained(save_dir)
#logger.info("Saved model to {}".format(save_dir))

# ---------------------> Loading saved gpt-j-6B-8bit model <------------------- #
logger.info("Getting GPT from saved")
#gpt = GPTJForCausalLM.from_pretrained("./saved_models_gpt-j-6B-8bit/gpt-j-6B",low_cpu_mem_usage=True)
gpt = GPTJForCausalLM.from_pretrained("/sc/arion/projects/EHR_ML/GPTJ-6B/gpt-j-6B-8bit",low_cpu_mem_usage=True)

#logger.info("adding specialized tokens PAD")
#num_added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#print("We have added", num_added_tokens, "tokens")

logger.info("resizing gpt token embeddings:")
gpt.resize_token_embeddings(len(tokenizer))
logger.info("resized token embeddings are: {}".format(gpt.resize_token_embeddings(len(tokenizer))))
#logger.info("resized token embeddings are: {}".format(gpt.resize_token_embeddings(num_added_tokens)))


print("is torch cuda available??")
print(torch.cuda.is_available())
logger.info("is cuda available??")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gpt.to(device)
logger.info("loaded from saved model: {}".format(gpt))
# ---------------------> Text generation example <------------------- #
prompt = tokenizer("A cat sat on a mat", return_tensors='pt')

prompt = {key: value.to(device) for key, value in prompt.items()}
out = gpt.generate(**prompt, min_length=128, max_length=128, do_sample=True)
logger.info("Generated text: {}".format(tokenizer.decode(out[0])))

print("*******print the modules and the named modules")
print(gpt.modules())
print(gpt.named_modules())


#print("printing module adapters")
#for module in gpt.modules():
#    print(module.adapter)


# ---------------------> LoRA fine-tuning example <------------------- #

def add_adapters2(model, adapter_dim=4, p = 0.1):
    assert adapter_dim > 0

    for name, module in model.named_modules():
      if isinstance(module, FrozenBNBLinear):
          if "attn" in name or "mlp" in name or "head" in name:
              print("Adding adapter to", name)
              module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Dropout(p=p),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
              print("Initializing", name)
              nn.init.zeros_(module.adapter[2].weight)

          else:
              print("Not adding adapter to", name)
      elif isinstance(module, FrozenBNBEmbedding):
          print("Adding adapter to", name)
          module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Dropout(p=p),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
          print("Initializing", name)
          nn.init.zeros_(module.adapter[2].weight)

#add_adapters(gpt)
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#gpt.to(device)




def add_adapters(model, adapter_dim=16):
    assert adapter_dim > 0

    for module in model.modules():
        print("printing module:")
        print(module)

        if isinstance(module, FrozenBNBLinear):
            module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            print("printing module adapters for FRozenBNBLinear")
            print(module.adapter)
            nn.init.zeros_(module.adapter[1].weight)
            ### PJ updated to match gustavocortal's repo
            #nn.init.zeros_(module.adapter[2].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            print("printing module adapters for FrozenBNBEmbedding")
            print(module.adapter)
            nn.init.zeros_(module.adapter[1].weight)
            ### PJ updated to match gustavocortal's repo
            #nn.init.zeros_(module.adapter[2].weight)

logger.info("adding adapters")
#add_adapters(gpt)
###PJ changed to follow gustavocortal's
#add_adapters2(gpt)

add_adapters2(gpt)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt.to(device)

gpt.gradient_checkpointing_enable()
logger.info("done adding adapters: {}".format(gpt))

print("loading dataset")
# example dataset
data = pd.read_csv('/sc/arion/projects/EHR_ML/VIZIEHR2023/gustavecortal_pgbp_example.csv')
data['sentence'] = 'Quote: ' + data['sentence']
train, test = train_test_split(data, test_size=0.01)

train.to_csv('/sc/arion/projects/EHR_ML/VIZIEHR2023/gustavecortal_train_pgbp_example.csv', index=False)
test.to_csv('/sc/arion/projects/EHR_ML/VIZIEHR2023/gustavecortal_test_pgbp_example.csv', index=False)

#dataset = load_dataset("/sc/arion/projects/EHR_ML/VIZIEHR2023/gustavecortal_pgbp_example.csv", streaming=True)
dataset = load_dataset('csv', data_files={'train': '/sc/arion/projects/EHR_ML/VIZIEHR2023/gustavecortal_train_pgbp_example.csv',
                                          'test': '/sc/arion/projects/EHR_ML/VIZIEHR2023/gustavecortal_test_pgbp_example.csv'})
logger.info("loading dataset: {}".format(dataset))


def tokenize_function(examples):
    #num_added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #print("We have added", num_added_tokens, "tokens")
    
    return tokenizer(examples["sentence"], padding=True, truncation=True, max_length= 128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets.set_format("torch")


from torch.utils.data import DataLoader

full_train_dataset = tokenized_datasets["train"]
train_dataloader = DataLoader(full_train_dataset, shuffle=False, batch_size=8)



# custom dataset
#dataset = load_dataset('text', data_files={'train': ['article-1.txt', 'article-2.txt'], 'test': ['article-3.txt', 'article-4.txt']})
print("setting optimizer: Adam 8-bit")
optimizer = Adam8bit(gpt.parameters(), lr=1e-5)
logger.info("setting optimizer: {}".format(optimizer))

# Set the model to training mode
start = time.time()
logger.info("Start time: {}".format(start))

# Training loop --- this is not gustavocortal's training loop.. lets see how this training function works!!
#with torch.cuda.amp.autocast():
#    for row in tqdm(dataset["train"]):
#        if len(row[""]) <= 1:
#            continue
#        batch = tokenizer(row["text"], truncation=True, max_length=128, return_tensors='pt')
#        batch = {k: v.cuda() for k, v in batch.items()}
#        out = gpt.forward(**batch,)
#        loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
#                               reduction='mean')
#        print(loss)
#        loss.backward()
#        optimizer.step()
#        optimizer.zero_grad()

logger.info("Starting training")
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
logger.info("Num training steps: {}".format(num_training_steps))

logger.info("getting lr scheduler")
lr_scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, int(num_training_steps*0.1), num_training_steps
)
logger.info("LR scheduler: {}".format(lr_scheduler))

logger.info("Starting scaler")
scaler = torch.cuda.amp.GradScaler()
logger.info("scaler: {}".format(scaler))


progress_bar = tqdm(range(num_training_steps))
logger.info("progress bar: {}".format(progress_bar))

gpt.train()
gpt.gradient_checkpointing_enable()
k = 0
print("starting train:")
logger.info("number of epochs: {}".format(num_epochs))


for epoch in range(num_epochs):
    for batch in train_dataloader:
        
        logger.info("batch is: {}".format(batch))

        k = k + 1
        if k % 500 == 0:
            logger.info("value of k is: {}".format(k))
            #print(k)
            state = {'k' : k, 'epoch': num_epochs, 'lr_scheduler': lr_scheduler.state_dict(), 'state_dict': gpt.state_dict(), 'optimizer': optimizer.state_dict()}
            logger.info("state is: {}".format(state))
            torch.save(state, filepath)
        
        
        logger.info("batch items is: {}".format(batch.items()))
        batch = {k: v.to(device) for k, v in batch.items()}
         
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logger.info("creating gpt.forward(**batch)")

            out = gpt.forward(**batch,)
            logger.info("out value is: {}".format(out))

            loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                                reduction='mean', label_smoothing=0.1)
          
        print(loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        progress_bar.update(1)

print("end train")


logger.info("Finished fine-tuning in {}".format(time.time() - start))

# --------------> Saving fine-tuned model <-----------------#
try:
    save_dir = "finetuned_gpt-j-8_bit/gpt-j-6B"
    os.makedirs(save_dir)
    gpt.save_pretrained(save_dir)
except Exception as e:
    #print("Error saving model: ", e)
    logger.info("Error saving model: {}".format(e))



###### gpt eval -----------------
logger.info("***** GPT EVAL: *****")

gpt.eval()
with torch.no_grad():
  prompt = tokenizer("Quote:", truncation=True, padding=True, max_length=128, return_tensors='pt')
  prompt = {key: value.to(device) for key, value in prompt.items()}
  out = gpt.generate(**prompt, max_length=128, top_k=50, top_p=0.9, temperature=1.0, do_sample=True, repetition_penalty = 1.2, num_beams=1)
  print(tokenizer.decode(out[0]))
