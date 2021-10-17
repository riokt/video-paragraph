import torch
import torch.nn as nn
# from modules.common import *
from modules.common_layer import *

# class EncoderLayer(nn.Module):
#   def __init__(self, d_model, heads, dropout=0.1, keyframes=False):
#     super().__init__()
#     self.norm_1 = Norm(d_model)
#     self.norm_2 = Norm(d_model)
#     self.attn = MultiHeadAttention(heads, d_model, dropout=dropout) 
#     self.ff = FeedForward(d_model, dropout=dropout)
#     self.dropout_1 = nn.Dropout(dropout)
#     self.dropout_2 = nn.Dropout(dropout)
#     self.keyframes = keyframes

#   def forward(self, x, mask):
#     x2 = self.norm_1(x)
#     x = x + self.dropout_1(self.attn(x2,x2,x2,mask)[0])
#     x2 = self.norm_2(x)
#     if self.keyframes:
#       select = self.dropout_2(torch.sigmoid(self.ff(x2)))
#       x = x * select
#       return x, select
#     else:
#       x = x + self.dropout_2(self.ff(x2))
#       return x, None


class Encoder(nn.Module):
  # def __init__(self, ft_dim, d_model, N, heads, dropout, keyframes=False):
  def __init__(self, embedding_size, hidden_size, num_layers, num_heads, layer_dropout, act=False, 
          max_length=100, input_dropout=0.0,attention_dropout=0.0, relu_dropout=0.0, use_mask=False, total_key_depth=128, total_value_depth=128, filter_size=128):
    super(Encoder, self).__init__()
    super().__init__()
    # self.N = N
    self.embed = nn.Linear(embedding_size, hidden_size)
    # self.pe = PositionalEncoder(d_model, dropout=dropout)
    # self.layers = get_clones(EncoderLayer(d_model, heads, dropout, keyframes), N)
    # self.norm = Norm(d_model)
    # self.keyframes = keyframes

    # def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
    #              filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, 
    #              attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False):
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
    #         use_mask: Set to True to turn on future value masking
    #     """
    
        
    self.timing_signal = gen_timing_signal(max_length, hidden_size)
    ## for t
    self.position_signal = gen_timing_signal(num_layers, hidden_size)

    self.num_layers = num_layers
    self.act = act
    params =(hidden_size, 
                total_key_depth or hidden_size,
                total_value_depth or hidden_size,
                filter_size, 
                num_heads, 
                gen_bias_mask(max_length) if use_mask else None,
                layer_dropout, 
                attention_dropout, 
                relu_dropout)

    self.proj_flag = False
    if(embedding_size == hidden_size):
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.proj_flag = True

    self.enc = EncoderLayer(*params)
    
    self.layer_norm = LayerNorm(hidden_size)
    self.input_dropout = nn.Dropout(input_dropout)
    if(self.act):
        self.act_fn = ACT_basic(hidden_size)

  # def forward(self, src, mask):
  def forward(self, inputs, mask):
    # x = self.embed(src)
    # x = self.pe(x)
    # for i in range(self.N):
    #   x, select = self.layers[i](x, mask)
      
    # if self.keyframes:
    #   # select key frame features
    #   select = select.mean(dim=-1, keepdim=True) * mask.transpose(-1, -2).float()
    #   org_frame = src * select
    #   return self.norm(x), org_frame, select.squeeze(-1)
    # else:
    #   return self.norm(x), None, None

    # Add input dropout
    x = self.input_dropout(inputs)

    if(self.proj_flag):
        # Project to hidden size
        x = self.embedding_proj(x)
    else:
        x = self.embed(x)

    if(self.act):
        x, (remainders,n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
        return x, (remainders,n_updates), None
    else:
        for l in range(self.num_layers):
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
            x = self.enc(x)
        return x, None, None

  def get_keyframes(self, src, mask):
    x = self.embed(src)
    x = self.pe(x)
    for i in range(self.N):
      x, select = self.layers[i](x, mask)
    select = select.mean(dim=-1, keepdim=True) * mask.transpose(-1, -2).float()
    select = select.squeeze(-1)
    thres = min(75, src.size(1))
    indices = select.topk(thres, 1)[1].sort()[0]
    x = torch.gather(x, 1, indices.unsqueeze(-1).expand(x.size(0),-1,x.size(-1)))
    mask = torch.gather(mask, 2, indices.unsqueeze(1).expand(x.size(0),1,-1))
    return self.norm(x), mask

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