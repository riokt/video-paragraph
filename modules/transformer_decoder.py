import torch
import torch.nn as nn
from modules.common_layer import *
import time


class Embedder(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embed = nn.Embedding(vocab_size, d_model)
  def forward(self, x):
    return self.embed(x)


# class DecoderLayer(nn.Module):
#   def __init__(self, d_model, heads, dropout=0.1):
#     super().__init__()
#     self.norm_1 = Norm(d_model)
#     self.norm_2 = Norm(d_model)
#     self.norm_3 = Norm(d_model)
#     self.dropout_1 = nn.Dropout(dropout)
#     self.dropout_2 = nn.Dropout(dropout)
#     self.dropout_3 = nn.Dropout(dropout)
#     self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
#     self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
#     self.ff = FeedForward(d_model, dropout=dropout)

#   def forward(self, x, e_outputs, src_mask, trg_mask, layer_cache=None):
#     x2 = self.norm_1(x)
#     x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask, layer_cache=layer_cache, attn_type='self')[0])
#     x2 = self.norm_2(x)
#     context, attn = self.attn_2(x2, e_outputs, e_outputs, src_mask, attn_type='context')
#     x = x + self.dropout_2(context)
#     x2 = self.norm_3(x)
#     x = x + self.dropout_3(self.ff(x2))
#     return x, attn
    
    
class Decoder(nn.Module):
  # def __init__(self, vocab_size, d_model, N, heads, dropout):
  def __init__(self, embedding_size, hidden_size, num_layers, num_heads, layer_dropout, act=False, 
          max_length=100, input_dropout=0.0,attention_dropout=0.0, relu_dropout=0.0, use_mask=False, total_key_depth=128, total_value_depth=128, filter_size=128):
    # super().__init__()
    # self.N = N
    # self.embed = Embedder(vocab_size, d_model)
    # self.pe = PositionalEncoder(d_model, dropout=dropout)
    # self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
    # self.norm = Norm(d_model)
    self.cache = None
    # def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
    #              filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, 
    #              attention_dropout=0.0, relu_dropout=0.0, act=False):
    #     """
    #     Parameters:
    #         embedding_size: Size of embeddings
    #         hidden_size: Hidden size
    #         num_layers: Total layers in the Encoder
    #         num_heads: Number of attention heads
    #         total_key_depth: Size of last dimension of keys. Must be divisible by num_head
    #         total_value_depth: Size of last dimension of values. Must be divisible by num_head
    #         output_depth: Size last dimension of the final output
    #         filter_size: Hidden size of the middle layer in FFN
    #         max_length: Max sequence length (required for timing signal)
    #         input_dropout: Dropout just after embedding
    #         layer_dropout: Dropout for each layer
    #         attention_dropout: Dropout probability after attention (Should be non-zero only during training)
    #         relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
    #     """
        
    super(Decoder, self).__init__()
    
    self.timing_signal = gen_timing_signal(max_length, hidden_size)
    self.position_signal = gen_timing_signal(num_layers, hidden_size)
    self.num_layers = num_layers
    self.act = act
    self.embed = Embedder(embedding_size, hidden_size)
    params =(hidden_size, 
              total_key_depth or hidden_size,
              total_value_depth or hidden_size,
              filter_size, 
              num_heads, 
              gen_bias_mask(max_length), # mandatory
              layer_dropout, 
              attention_dropout, 
              relu_dropout)

    self.proj_flag = False
    print(embedding_size,hidden_size, "HOHO")
    if(embedding_size == hidden_size):
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.proj_flag = True
    self.dec = DecoderLayer(*params) 
    
    self.layer_norm = LayerNorm(hidden_size)
    self.input_dropout = nn.Dropout(input_dropout)
    if(self.act):
        self.act_fn = ACT_basic(hidden_size)
  
  # def _init_cache(self):
  #   self.cache = {}
  #   for i in range(self.N):
  #     self.cache['layer_%d'%i] = {
  #       'self_keys': None,
  #       'self_values': None,
  #     }    

  # def forward(self, trg, e_outputs, src_mask, trg_mask, step=None):
  def forward(self, inputs, encoder_output, src_mask, trg_mask, step=None):
    # if step == 1:
    #   self._init_cache()

    # x = self.embed(trg)
    # x = self.pe(x, step)
    # attn_w = []
    # for i in range(self.N):
    #   layer_cache = self.cache['layer_%d'%i] if step is not None else None
    #   x, attn = self.layers[i](x, e_outputs, src_mask, trg_mask, layer_cache=layer_cache)
    #   attn_w.append(attn)
    # return self.norm(x), sum(attn_w)/self.N
# def forward(self, inputs, encoder_output):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        if(self.proj_flag):
            # Project to hidden size
            x = self.embedding_proj(x)
        else:
            x = self.embed(x)
        
        if(self.act):
            x, (remainders,n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output)
            return x, (remainders,n_updates), None
        else:
            for l in range(self.num_layers):
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                x += self.position_signal[:, l, step].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                x, attn, _ = self.dec((x, encoder_output), src_mask, trg_mask)
        return x, attn, None

class ACT_basic(nn.Module):
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if(encoder_output):
                state, _ = fn((state,encoder_output))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        return previous_state, (remainders,n_updates)