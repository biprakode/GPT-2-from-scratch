import torch
from torch import nn
from model.ModelConfig import ModelConfig
from model.PositionalEmbedding import PositionalEmbedding
from model.TrainingConfig import TrainingConfig
from tokenizer.bpe import BPETokenizer
from model.attention import MultiHeadAttention
from model.layernorm import LayerNorm
from model.feedforward import FeedForward
from model.block import TransformerBlock
from torch import Tensor
import torch.nn.functional as F

class GPT2(nn.Module):
    def __init__(self , modelconfig:ModelConfig , trainingconfig:TrainingConfig):
        super().__init__()
        self.wte = nn.Embedding(modelconfig.vocab_size , modelconfig.n_embd)
        self.wpe = PositionalEmbedding(modelconfig)
        self.drop = nn.Dropout(modelconfig.embd_pdrop) # applied to- token_emb + pos_emb
        self.h = nn.ModuleList([TransformerBlock(modelconfig , trainingconfig) for _ in range(modelconfig.n_layer)])
        self.ln_f = LayerNorm(modelconfig)
        self.lm_head = nn.Linear(modelconfig.n_embd , modelconfig.vocab_size , bias=False)

        self.wte.weight = self.lm_head.weight # weight tying (token -> vector -> token)

        self.config = modelconfig
        self.apply(self._init_weights)

        self._cache_initialized = False
        self.eos_token_id = 50256

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self , input_ids , position_ids=None , past_key_values=None , use_cache=False):
        batch_size , seq_len = input_ids.size()

        if position_ids is None: # continue position from where left off
            if use_cache and self._cache_initialized:
                start_pos = self.h[0].attn.current_pos # first layer position
                position_ids = torch.arange(start_pos , start_pos + seq_len , dtype=torch.long, device=input_ids.device)
            else:
                position_ids = torch.arange(0, seq_len, dtype=torch.long , device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)  # (1, seq_len)

        token_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)

        hidden_state = token_emb + pos_emb
        hidden_state = self.drop(hidden_state)
        for block in self.h:
            hidden_state = block(hidden_state , use_cache=use_cache)

        hidden = self.ln_f(hidden_state)
        logits = self.lm_head(hidden) # batch , seq_len , vocab_size
        if use_cache:
            self._cache_initialized = True

        return logits

    def generate(self , prompt_ids , max_new_tokens=50 , temperature=1.0 , top_k=None , top_p=.9):
        self.eval()
        self.reset_cache()
        device = next(self.parameters()).device
        prompt_ids = prompt_ids.to(device)
        generated = prompt_ids.clone()
        with torch.no_grad():
            logits = self.forward(generated , use_cache=True)
            last_logit = logits[:, -1, :] # (batch , vocab_size)
            new_token = self._sample(last_logit , temperature=temperature, top_k=top_k, top_p=top_p , prev_tokens=generated)
            generated = torch.cat([generated , new_token], dim = 1)

            if self.eos_token_id is not None and (new_token == self.eos_token_id).all():
                return generated

            for _ in range(max_new_tokens-1):
                logits = self.forward(new_token , use_cache=True)
                last_logit = logits[:, -1, :]
                new_token = self._sample(last_logit, temperature=temperature, top_k=top_k, top_p=top_p , prev_tokens=generated)
                generated = torch.cat([generated , new_token], dim = 1)

                if self.eos_token_id is not None and (new_token == self.eos_token_id).all():
                    break

        return generated

        pass

    def _sample(self , logits , temperature , top_k , top_p , prev_tokens , repetition_penalty=1.2):
        if repetition_penalty != 1.0 and prev_tokens is not None:
            for i in range(logits.shape[0]):
                prev_token_ids = torch.unique(prev_tokens[i])
                score = logits[i , prev_token_ids] # logits of prev tokens
                logits[i , prev_token_ids] = torch.where(score > 0 , score / repetition_penalty , score * repetition_penalty)

        logits = logits / temperature

        #topK filtering
        if top_k is not None:
            top_k = min(top_k , logits.size(-1))
            # torch.topk returns (values, indices) sorted descending
            # [0] gets values, [..., -1, None] gets last (k-th) value and adds dimension
            to_remove = logits < torch.topk(logits , top_k)[0][..., -1 , None]
            logits[to_remove] = float('-inf')

        #topP sampling
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits , descending = True) # sorted_indices: (batch, vocab_size) - original positions of sorted logits
            sorted_probs = F.softmax(sorted_logits , dim = -1) # Convert sorted logits to probabilities
            cumulative_probs = torch.cumsum(sorted_probs ,dim = -1) # cumulative_probs[i] = sum of top (i+1) probabilities
            sorted_ind_to_remove = cumulative_probs > top_p

            # Shift mask right by 1 to keep at least the top token
            # (Even if top token alone exceeds p, we keep it)
            sorted_ind_to_remove[..., 1:] = sorted_ind_to_remove[..., :-1].clone()
            sorted_ind_to_remove[..., 0] = False  # Always keep the top token

            # Unsort the mask back to original logit positions
            # scatter: put sorted mask values back to original positions
            ind_to_remove = sorted_ind_to_remove.scatter(1 , sorted_indices , sorted_ind_to_remove)
            logits[ind_to_remove] = -float('inf')

        probs = F.softmax(logits , dim = -1)
        next_token = torch.multinomial(probs , 1)
        return next_token


    def reset_cache(self):
        for block in self.h:
            block.reset_cache()
        self._cache_initialized = False


