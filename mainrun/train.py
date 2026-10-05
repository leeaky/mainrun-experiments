import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

from config import Hyperparameters, load_config, warmup_cosine_factor, wsd_factor
from muon import Muon, CombinedOptimizer
from ema import EMA

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    scaled_residual_init: bool = False
    attn_impl: str = "manual"
    mlp_kind: str = "gelu"
    qk_norm: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.attn_impl = cfg.attn_impl
        self.dropout_p = cfg.dropout
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)


class SwiGLUMLP(nn.Module):
    """Gated-linear-unit MLP, param-matched to the 4x GELU MLP: 3 bias-free
    matrices of hidden width h ~= 8/3 * d_model (rounded to a multiple of 8)
    total ~= the GELU MLP's 2 * 4d matrices."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        h = round(8 * cfg.d_model / 3 / 8) * 8
        self.w_gate = nn.Linear(cfg.d_model, h, bias=False)
        self.w_up   = nn.Linear(cfg.d_model, h, bias=False)
        self.w_down = nn.Linear(h, cfg.d_model, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = SwiGLUMLP(cfg) if cfg.mlp_kind == "swiglu" else MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        if cfg.scaled_residual_init:
            std = 0.02 / math.sqrt(2 * cfg.n_layer)
            for block in self.blocks:
                nn.init.normal_(block.attn.proj.weight, mean=0.0, std=std)
                mlp_out = block.mlp.net[2] if hasattr(block.mlp, "net") else block.mlp.w_down
                nn.init.normal_(mlp_out.weight, mean=0.0, std=std)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

def split_params_for_muon(model):
    """Partition model.named_parameters() into (muon_params, adamw_params).

    Muon must only see 2D *hidden* weight matrices. Everything else - the
    token embedding (tied to the LM head, so the same tensor), the 3D
    positional-embedding parameter, every LayerNorm weight/bias, and every
    Linear bias - belongs on AdamW.

    Standard hybrid-Muon rule (no deviation): a param is a Muon param iff it
    is a 2D tensor that is not the token embedding. That covers the fused
    qkv, the attention output proj, and both MLP matrices per block. The 3D
    pos_emb, every 1D LayerNorm weight/bias, and every Linear bias fall
    through to AdamW; the tied token_emb/head weight is yielded once (as
    'token_emb.weight') and routed to AdamW by the name check.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "token_emb" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def main():
    args = load_config(os.environ.get("MAINRUN_CONFIG"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok_corpus = train_titles if args.tokenizer_scope == "train_only" else train_titles + val_titles
    tok = BPETokenizer(train_tokenizer(tok_corpus, args.vocab_size, eos_token=eos_token))
    if args.tokenizer_scope == "train_only":
        tok_tv = train_tokenizer(train_titles + val_titles, args.vocab_size, eos_token=eos_token)
        va = set(tok.tk.get_vocab().keys())
        vb = set(tok_tv.get_vocab().keys())
        logger.log("tokenizer_overlap_info",
                   vocab_overlap_frac=len(va & vb) / len(va | vb),
                   train_only_vocab_size=len(va),
                   train_and_val_vocab_size=len(vb))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    if args.shuffle_batches:
        n_chunks = len(train_ids) // (args.block_size + 1)
        train_chunks = train_ids[:n_chunks * (args.block_size + 1)].view(n_chunks, args.block_size + 1)
        batches = n_chunks // args.batch_size
    else:
        batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    val_title_token_counts = [len(tok.encode(t)) for t in val_titles]
    avg_tokens_per_val_title = sum(val_title_token_counts) / len(val_title_token_counts)
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size,
               max_steps=max_steps,
               avg_tokens_per_val_title=avg_tokens_per_val_title)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        scaled_residual_init = args.scaled_residual_init,
        attn_impl  = args.attn_impl,
        mlp_kind   = args.mlp_kind,
        qk_norm    = args.qk_norm,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    ema = EMA(model.named_parameters(), decay=args.ema_decay)
    if args.ema_decay > 0:
        logger.log("ema_info", ema_decay=args.ema_decay)
    
    def _sched_fn(step):
        if args.lr_schedule == "wsd":
            return wsd_factor(step, max_steps, args.warmup_frac,
                              args.lr_floor_frac, args.stable_frac,
                              shape=args.wsd_decay_shape)
        return warmup_cosine_factor(step, max_steps, args.warmup_frac,
                                    args.lr_floor_frac)

    if args.optimizer == "adamw":
        decay = [p for _, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(args.beta1, args.beta2))
        logger.log("optimizer_info",
                   decay_tensors=len(decay),
                   no_decay_tensors=len(no_decay),
                   decay_params=sum(p.numel() for p in decay),
                   no_decay_params=sum(p.numel() for p in no_decay))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _sched_fn)
    elif args.optimizer == "muon":
        muon_params, adamw_params = split_params_for_muon(model)
        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_muon + n_adamw == n_total, \
            f"param split lost/duplicated tensors: {n_muon}+{n_adamw} != {n_total}"
        muon_opt = Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum,
                        ns_steps=args.muon_ns_steps)
        aw_decay = [p for p in adamw_params if p.ndim >= 2]
        aw_no_decay = [p for p in adamw_params if p.ndim < 2]
        aw_wd = args.weight_decay if args.weight_decay_embeddings else 0.0
        adamw_opt = torch.optim.AdamW(
            [{"params": aw_decay, "weight_decay": aw_wd},
             {"params": aw_no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(args.beta1, args.beta2))
        opt = CombinedOptimizer([muon_opt, adamw_opt], sched_fn=_sched_fn)
        scheduler = None
        logger.log("optimizer_info", optimizer="muon",
                   muon_tensors=len(muon_params), muon_params=n_muon,
                   adamw_decay_tensors=len(aw_decay),
                   adamw_no_decay_tensors=len(aw_no_decay),
                   adamw_params=n_adamw)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    last_train_loss = None
    best_val_loss = float("inf")
    best_val_step = 0
    final_val_loss = None
    for epoch in range(1, args.epochs + 1):
        if args.shuffle_batches:
            order = list(range(n_chunks))
            random.Random(args.seed + epoch).shuffle(order)
        for b in tqdm(range(batches), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            if args.shuffle_batches:
                sel = torch.tensor(order[b * args.batch_size:(b + 1) * args.batch_size], dtype=torch.long)
                rows = train_chunks[sel]
                xb = rows[:, :-1].to(device)
                yb = rows[:, 1:].to(device)
            else:
                xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            if args.use_amp:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(xb, yb)
            else:
                _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            else:
                opt.apply_schedule(step)   # CombinedOptimizer: step is 1-based
            ema.update(model.named_parameters(), step)

            last_train_loss = loss.item()
            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=last_train_loss,
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                with ema.swapped_in(model):
                    val_loss = evaluate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                final_val_loss = val_loss
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          train_loss=last_train_loss,
                          elapsed_time=elapsed)

    raw_final_val_loss = evaluate() if args.ema_decay > 0 else final_val_loss
    logger.log("training_summary",
               best_val_loss=best_val_loss,
               best_val_step=best_val_step,
               final_val_loss=final_val_loss,
               raw_final_val_loss=raw_final_val_loss)

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
