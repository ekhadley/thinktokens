from utils import *

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
if MAIN: print(bold, red, underline, f"device set to {device}", endc)

@dataclass
class modelConfig:
    d_model: int = 768
    d_vocab: int = 50304 # 50257
    init_range: float = 0.02
    n_ctx: int = 512
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

@dataclass
class trainingConfig:
    train_tokens = 100_000
    test_tokens = 5_000
    batch_size = 8
    epochs = 3
    lr = 1e-5
    weight_decay = 1e-3
    wandb_project: Optional[str] = "normalformer"
    wandb_name: Optional[str] = None

class attention(nn.Module):
    def __init__(self, cfg: modelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model, 3*cfg.d_head*cfg.n_heads)
        self.out = nn.Linear(cfg.d_head, cfg.d_model)

    def forward(self, x: t.Tensor):
        b, seq, _ = x.shape
        qkv = self.proj(x)
        q, k, v = qkv.split(self.cfg.d_model, dim=-1)
        q = q.view(b, seq, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        k = k.view(b, seq, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        v = v.view(b, seq, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)

        values = nn.F.scaled_dot_product_attention(q, k, v, is_causal=True) # ugh

        values = values.transpose(1, 2).contiguous().view(x.shape)
        out = self.out(values)
        return out


class tblock(nn.Module):
    def __init__(self, cfg: modelConfig, targs: trainingConfig):
        super().__init__()
        self.cfg, self.targs = cfg, targs
        self.ln1 = nn.LayerNorm((cfg.d_model))
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True, )
        self.ln2 = nn.LayerNorm((cfg.d_model))
        self.mlp1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x: t.Tensor):
        normed = self.ln1(x)
        seq = x.shape[1]
        mask = t.triu(t.ones((seq, seq), device=device), diagonal=1).bool()
        attn_out, _ = self.attn(normed, normed, normed, is_causal=True, attn_mask=mask)
        
        post_attn = x + attn_out
        mlp_out = self.ln2(post_attn)
        mlp_out = self.act(self.mlp1(mlp_out))
        mlp_out = self.mlp2(mlp_out)
        post_mlp = post_attn + mlp_out
        return post_mlp

class gpt2(nn.Module):
    def __init__(self, cfg, targs):
        super().__init__()
        self.cfg, self.targs = cfg, targs

        self.E = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.PE = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.Sequential(*[tblock(cfg, targs) for i in range(cfg.n_layers)])
        self.UE = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        self.ln_final = nn.LayerNorm((cfg.d_model))

        self.to(device)
        self.device = device
        
        self.tk = GPT2TokenizerFast.from_pretrained('gpt2')
        self.opt = t.optim.AdamW(self.parameters(), targs.lr, betas=(0.9, 0.95), weight_decay=targs.weight_decay, fused=True)
        self.scheduler = t.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=1e4, eta_min=0)

    def forward(self, x: t.Tensor):
        if isinstance(x, str): x = self.tokenize(x)['input_ids'].to(device)
        resid = self.E(x) + self.PE(t.arange(0, x.shape[-1], device=device, requires_grad=False).unsqueeze(0))
        resid = self.blocks(resid)
        logits = self.UE(self.ln_final(resid))
        return logits

    def loss(self, logits: t.Tensor, labels: t.tensor):
        logprobs = logits.log_softmax(dim=-1)
        correct_logprobs = logprobs[:, :-1].gather(dim=-1, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        return -(correct_logprobs.mean())

    def trainstep(self, loss: t.Tensor):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

    def tokenize(self, prompt): return self.tk(prompt, return_tensors='pt')
    def decode(self, tokens): return self.tk.batch_decode(tokens)
    def accuracy(self, logits: t.Tensor, tokens: t.Tensor):
        preds = logits.squeeze().argmax(dim=-1)
        return (preds[...,:-1]==tokens.squeeze()[...,1:]).float().mean()
    def yap(self, _prompt, ntok=30, show=False):
        out = _prompt
        prompt = self.tokenize(_prompt)['input_ids'].to(device).squeeze()
        for i in range(ntok):
            logits = self.forward(prompt).squeeze()
            nexttok = self.sample(logits)
            prompt = t.cat([prompt, t.tensor([nexttok], device=device)], dim=-1)
            out += self.tk.decode(nexttok)
            if show:
                print(out)
                print()
        return out
    def sample(self, logits: t.Tensor, k=5, temp=0):
        vals, indices = logits.squeeze()[-1].topk(k)
        if temp != 0: vals /= temp
        idx = t.distributions.categorical.Categorical(logits=vals).sample().item()
        return indices[idx].item()
    def log_completion(self, completion):
        table = wandb.Table(data=[[completion]], columns=['completion'])
        wandb.log({"completion": table})
    def load(self, path):
        self.load_state_dict(t.load(path).state_dict())


def train(cfg: modelConfig, targs: trainingConfig, load=None):
    n_train_iter, n_test_iter = targs.train_tokens//targs.batch_size, targs.test_tokens//targs.batch_size
    #t.set_float32_matmul_precision('high')
    model = gpt2(cfg, targs)
    if load is not None: model.load(load)

    print(bold, purple, f"model of {sum(param.numel() for param in model.parameters()):,} params created on {device}", endc)

    #dataset_title, dataset_name  = "wikitext",  "wikitext-103-raw-v1"
    dataset_title, dataset_name  = "HuggingFaceFW/fineweb-edu", "sample-10BT"
    print(f"{yellow}loading raw dataset: {dataset_title} ({dataset_name}){endc}")
    dataset = datasets.load_dataset(dataset_title, name=dataset_name, split="train", streaming=True).map(dataset_tokenize_func, batched=True) 
    trainloader = DataLoader(dataset, batch_size=targs.batch_size)
    testloader = DataLoader(dataset, batch_size=targs.batch_size)
    
    print(f"{yellow}train and test dataloaders created{endc}")

    wandb.init(project=targs.wandb_project, name=targs.wandb_name)
    wandb.watch(model)

    nbatch = 0
    for epoch in range(targs.epochs):
        testloss, testacc = 0, 0
        with t.inference_mode():
            testiter, testrange = iter(testloader), trange(n_test_iter, ncols=120, desc="testing. . .")
            for i in testrange:
                toks = next(testiter)['input_ids'].to(device)
                #with t.autocast(device_type='cuda'):
                logits = model.forward(toks)
                testloss += model.loss(logits, toks).item()
                testacc += model.accuracy(logits, toks).item()
            testloss /= n_test_iter
            testacc /= n_test_iter
        wandb.log({"testloss": testloss, "testacc": testacc})

        print(yellow, yap := model.yap("George Washington was", show=False), endc)
        model.log_completion(yap)

        model.train()
        trainiter, trainrange = iter(trainloader), trange(n_train_iter, ncols=120)
        for i in trainrange:
            toks = next(trainiter)['input_ids'].to(device)
            #with t.autocast(device_type='cuda'):
            logits = model.forward(toks)
            loss = model.loss(logits, toks)
            model.trainstep(loss)
            nbatch += 1
            if i % 10 == 0:
                wandb.log({"trainloss": loss})
                trainrange.set_description(f"{bold+purple}[{epoch}/{targs.epochs}] {blue}loss:{loss:.4f} acc:{testacc:.6f} testloss:{testloss:.6f}")
        t.save(model, f"D:\\wgmn\\think\\saves\\gpt2s_e{epoch}.pth")

    wandb.finish()

    model.eval()
    return model

if MAIN:
    #t.manual_seed(0)
    cfg = modelConfig()
    targs = trainingConfig()

    #model = gpt2(cfg, targs)
    #t.save(model, f"D:\\wgmn\\think\\saves\\gpt2.pth")
    
    model = train(cfg, targs)