from utils import *
import train_normal
MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
if MAIN: print(bold, red, underline, f"device set to {device}", endc)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

@dataclass
class modelConfig:
    d_model: int = 512
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 512
    d_head: int = 64
    d_mlp: int = 2048
    n_heads: int = 8
    n_layers: int = 8

    num_think_tokens = 5039 # (50257 + 5039) % 64 = 0

@dataclass
class trainingConfig:
    train_tokens = 100_000
    test_tokens = 5_000
    batch_size = 8
    epochs = 10
    lr = 3e-4
    rl_weight: float = 1.0
    weight_decay = 3e-3
    wandb_project: Optional[str] = "thinkformer"
    wandb_name: Optional[str] = None

class tblock(nn.Module):
    def __init__(self, cfg: modelConfig, targs: trainingConfig):
        super().__init__()
        self.cfg, self.targs = cfg, targs
        self.ln1 = nn.LayerNorm((cfg.d_model))
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
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

class gpt2t(nn.Module):
    def __init__(self, cfg, targs):
        super().__init__()
        cfg.d_full_vocab = cfg.d_vocab + cfg.num_think_tokens
        self.cfg, self.targs = cfg, targs

        self.E = nn.Embedding(self.cfg.d_full_vocab, cfg.d_model)

        self.PE = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.Sequential(*[tblock(cfg, targs) for i in range(cfg.n_layers)])
        self.UE = nn.Linear(cfg.d_model, cfg.d_full_vocab, bias=False)
        self.ln_final = nn.LayerNorm((cfg.d_model))

        self.pad = cfg.d_vocab

        self.to(device)
        self.device = device

        self.tk = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tk.add_special_tokens({'pad_token': '<PAD>'})
        self.opt = t.optim.AdamW(self.parameters(), targs.lr, betas=(0.9, 0.95), weight_decay=targs.weight_decay, fused=(self.device==t.device('cuda')))


    def forward(self, x: t.Tensor, last=True) -> t.Tensor:
        #if isinstance(x, str): x = self.tokenize(x)['input_ids'].to(device)
        if x.ndim < 2: x = x.unsqueeze(0)
        resid = self.E(x) + self.PE(t.arange(0, x.shape[-1], device=device, requires_grad=False).unsqueeze(0))
        resid = self.blocks(resid)
        resid_final = self.ln_final(resid)
        if last: return self.UE(resid_final[:,-1,:]).squeeze()
        else: return self.UE(resid_final)
    
    @t.inference_mode()
    def batch_forward(self, tokens: t.Tensor):
        batch_size, seq_len = tokens.shape
        n_real_tok = (tokens != self.pad).count_nonzero(dim=-1) - 1
        not_done = t.ones((batch_size), device=device, dtype=t.bool)
        batch_indices = t.arange(batch_size, device=device, requires_grad=False, dtype=t.int32)
        # ctx is a storage tensor for the sequences interspersed with the model's thought tokens.
        ctx = t.full((batch_size, self.cfg.n_ctx), self.cfg.d_vocab, device=device)# final entry in the original vocab is <PAD>
        ctx[:,0] = tokens[:,0] # set the first token in ctx to the first tokens
        endpts = t.ones((batch_size), device=device, dtype=t.int32) # keeps track of how many tokens (think + normal) are in the ctx for each elem in batch
        for i in range(1, tokens.shape[-1]): # iterate over the sequence dimension of each sequence in the batch
            lastlogits = self.forward(ctx[:,:endpts.max()], last=True) # get logits for last sequence position for all tokens up to current seq position
            lastpreds = lastlogits.argmax(dim=-1) # get next token predictions
            while (tmask := t.logical_and((lastpreds > self.cfg.d_vocab).squeeze(), not_done)).any() and endpts.max() < self.cfg.n_ctx - 2:
                ctx[tmask, endpts[tmask]] = lastpreds[tmask]
                endpts[tmask] += 1

                lastlogits = self.forward(ctx[tmask][:, :endpts.max()], last=True)
                lastpreds[tmask] = lastlogits.argmax(dim=-1)

            ctx[batch_indices, endpts] = tokens[batch_indices, i]
            not_done = endpts+1 < n_real_tok
            endpts[not_done] += 1
            if endpts.max() == self.cfg.n_ctx - 1: return ctx
            if (~not_done).all(): return ctx
        return ctx

    def loss(self, context: t.Tensor, tokens: t.Tensor, logprobs: t.Tensor, nothink_logprobs: t.Tensor, return_all=False) -> t.Tensor | Tuple[t.Tensor]:
        assert context.ndim == 1
        #self.tokprint(context)
        #print(context)
        
        with t.no_grad():
            #tmask = context > self.cfg.d_vocab # mask of which tokens are thinking
            tmask = t.gt(context, self.cfg.d_vocab) # mask of which tokens are thinking
            npmask = context != self.pad
            nmask = t.logical_and(~tmask, npmask) # normal tokens are not thinking tokens nor padding tokens
            n_normal_tok = nmask.count_nonzero()
            
            print(red, f"think:{tmask.count_nonzero().item():.3f}, {blue}normal:{nmask.count_nonzero().item():.3f}, {green}pad:{(context == self.pad).count_nonzero().item():.3f}", endc)
            nothink_logprobs = nothink_logprobs[:n_normal_tok]
            
            seq_idx = t.arange(0, len(tokens), 1, device=device)
        
        normal_logprobs, think_logprobs = logprobs[nmask], logprobs[tmask]
        
        supervised_logprobs = -normal_logprobs[:-1][seq_idx[:n_normal_tok-1], tokens[1:n_normal_tok]].squeeze()
        supervised_loss = supervised_logprobs.mean()
        
        if tmask.any():
            nothink_supervised_logprobs = -nothink_logprobs[:-1][seq_idx[:n_normal_tok-1], tokens[1:n_normal_tok]].squeeze()

            rewards = t.zeros((npmask.sum()), device=device, requires_grad=False)
            rewards[t.where(nmask)[0].squeeze()[:-1]] = (supervised_logprobs - nothink_supervised_logprobs)
            #diff = supervised_logprobs - nothink_supervised_logprobs
            #print(cyan, supervised_logprobs.mean(), purple, nothink_supervised_logprobs.mean(), endc)
            #print(orange, diff.mean(), lime, diff.sum(), endc)
            #print(yellow, (diff > 0).float().mean(), green, (diff < 0).float().mean(), endc)
            #print(red, rewards, endc)
            #print(orange, rewards.mean(), lime, rewards.sum(), endc)
            #print(yellow, (rewards > 0).float().mean(), green, (rewards < 0).float().mean(), endc)
            rtg = rewards.flip(0).cumsum(dim=0).flip(0)
            #print(blue, rtg, cyan, rtg.mean(), endc)
            #exit()
            tmask_where = t.where(tmask)[0].squeeze()
            #nmask_where = t.where(nmask)[0].squeeze()
            #print(cyan, 'nmask_where:', nmask_where, endc)
            #print(purple, 'tmask_where', tmask_where, endc)
            #print(context)
            #print(context[tmask_where])
            #print(f"{yellow}{rtg.shape=}{purple}{think_logprobs.shape=}, {red}{seq_idx.shape=}, {orange}{len(tmask_where)=}, {pink}{context[tmask_where].shape=}{endc}")
            if False: #tmask_where[-1] >= nmask[-1]:
                print("............1...............")
                rtg[tmask_where]
                time.sleep(1)
                print("............2...............")
                think_logprobs[seq_idx[:len(tmask_where)], context[tmask_where]]
                time.sleep(1)
                print("............3...............")
                seq_idx[:len(tmask_where)]
                time.sleep(1)
                print("............4...............")
                context[tmask_where]
                time.sleep(1)
                print("............5...............")
                time.sleep(1)

            rl_loss = (rtg[tmask_where] * -think_logprobs[seq_idx[:tmask_where.numel()], context[tmask_where]]).mean()
        else:
            rl_loss = t.tensor(0, device=device, dtype=t.float32)

        loss = supervised_loss + rl_loss*self.targs.rl_weight
        if return_all: return loss, supervised_loss, rl_loss
        return loss

    def batched_loss(self, context: t.Tensor, tokens: t.Tensor):
        batch_size = context.shape[0]
        all_logits = self.forward(context, last=False).squeeze()
        all_logprobs = all_logits.log_softmax(dim=-1)
        with t.inference_mode():
            all_nothink_logits = self.forward(tokens, last=False).squeeze()
            all_nothink_logprobs = all_nothink_logits[...,:self.cfg.d_vocab].log_softmax(dim=-1).squeeze()
        losses, sup_losses, rl_losses = [], [], []
        for i, c in enumerate(context):
            loss, sup_loss, rl_loss = self.loss(c, tokens[i], all_logprobs[i], all_nothink_logprobs[i], return_all=True)
            losses.append(loss)
            sup_losses.append(sup_loss)
            rl_losses.append(rl_loss)
        
        return sum(losses)/batch_size, sum(sup_losses)/batch_size, sum(rl_losses)/batch_size

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
            prompt = t.cat([prompt, nexttok.unsqueeze(0)], dim=-1)
            out += self.tk.decode(nexttok)
            if show:
                self.tokprint()
                print()
        self.tokprint(prompt)
        return out
    def sample(self, logits: t.Tensor, k=5, temp=0):
        #vals, indices = logits.squeeze()[-1].topk(k)
        vals, indices = logits.squeeze().topk(k)
        if temp != 0: vals /= temp
        idx = t.distributions.categorical.Categorical(logits=vals).sample().item()
        return indices[idx]
    def log_completion(self, completion):
        table = wandb.Table(data=[[completion]], columns=['completion'])
        wandb.log({"completion": table})
    def load(self, path):
        self.load_state_dict(t.load(path).state_dict())
    def tokprint(self, _toks: t.Tensor, normal_color=cyan, think_color=lime, lim=None):
        toks = _toks.squeeze() if isinstance(_toks, t.Tensor) else t.tensor(_toks).squeeze()
        if toks.ndim > 1:
            for i in range(len(toks)):
                self.tokprint(toks[i])
            return
        toks = toks[:(lim if lim is None else -lim)]

        think_idx = toks > self.cfg.d_vocab
        out = ""
        for i, tok in enumerate(toks):
            if think_idx[i]:
                out += think_color + self.tk.decode(tok - self.cfg.d_vocab)
            else:
                out += normal_color + self.tk.decode(tok)
        out += endc
        print(out)

def train(cfg: modelConfig, targs: trainingConfig, load=None):
    n_train_iter, n_test_iter = targs.train_tokens//targs.batch_size, targs.test_tokens//targs.batch_size
    #t.set_float32_matmul_precision('high')

    model = gpt2t(cfg, targs)
    if load is not None: model.load(load)

    print(bold, purple, f"model of {sum(param.numel() for param in model.parameters()):,} params created on {device}", endc)

    #dataset_title, dataset_name  = "wikitext",  "wikitext-103-raw-v1"
    dataset_title, dataset_name  = "HuggingFaceFW/fineweb-edu", "sample-10BT"
    print(f"{yellow}loading raw dataset: {dataset_title} ({dataset_name}){endc}")
    dataset = datasets.load_dataset(dataset_title, name=dataset_name, split="train")
    dataset.set_format('torch')
    trainset, testset = dataset.train_test_split(train_size=1000, test_size=100, shuffle=True).values()

    tokenize_func = functools.partial(dataset_tokenize_func, max_length=model.cfg.n_ctx//2)
    tokenized_trainset = trainset.map(tokenize_func, batched=True, remove_columns=trainset.column_names)
    tokenized_testset = testset.map(tokenize_func, batched=True, remove_columns=testset.column_names)
    trainloader = DataLoader(tokenized_trainset, batch_size=targs.batch_size, shuffle=True)
    testloader = DataLoader(tokenized_testset, batch_size=targs.batch_size, shuffle=True)
    
    print(f"{yellow}train and test dataloaders created{endc}")

    wandb.init(project=targs.wandb_project, name=targs.wandb_name)
    wandb.watch(model)

    for epoch in range(targs.epochs):
        #tl = tqdm(testloader, ncols=150)
        #testloss, testacc = 0, 0
        #model.eval()
        #with t.inference_mode():
        #    for batch in tl:
        #        toks = batch['tokens'].to(device)
        #        logits = model(toks)
        #        testloss += model.loss(logits, toks).item()
        #        testacc += model.accuracy(logits, toks).item()
        #    testloss /= len(tl)
        #    testacc /= len(tl)
        #wandb.log({"testloss": testloss, "testacc": testacc})

        #print(yellow, yap := model.yap("George Washington was", show=False), endc)
        model.yap("George Washington was")
        #model.log_completion(yap)
    

        trainrange = tqdm(trainloader)
        for batch in trainrange:
            tokens = batch['input_ids'].to(device)
            output_ctx = model.batch_forward(tokens).clone()
            model.opt.zero_grad()
            loss, supervised_loss, rl_loss = model.batched_loss(output_ctx, tokens)
            loss.backward()
            model.opt.step()

            trainrange.set_description(f"{bold+purple}[{epoch}/{targs.epochs}] {blue}losses: [sup:{supervised_loss.item():.3f} rl:{rl_loss.item():.3f} total:{loss.item():.3f}]")
            wandb.log({"trainloss": loss})
            
            #del loss
            t.cuda.empty_cache()
        
        t.save(model, f"D:\\wgmn\\think\\saves\\gpt2s_think_e{epoch}.pth")

    wandb.finish()
    model.eval()
    return model

if MAIN:
    #t.manual_seed(0)
    cfg = modelConfig()
    targs = trainingConfig()

    model = train(cfg, targs)

    #model = gpt2t(cfg, targs)


# NOTE: high level approach: given a sequence, we use rl to let the model learn how to add
# hidden 'thinking' tokens to the context such that the next token prediction accuracy of
# later 'real' tokens goes up.
#    Curent concrete implementation: our model's dictionary grows. We train thinking tokens via RL
#  and we train normal tokens via supervision. The loss on the normal tokens is the usual crossentropy.
# The reward for the thinking tokens is the negative of the supervision loss on all following normal
# tokens. This encourages the model to produce thinking tokens that caused later next word predictions
# to become more accurate.
#    Training methodology: We treat next word prediction as a time-sequence rl task. We take a sample
# of N tokens in length. We give the model the first n tokens, and collect its prediction for the next
# token. If this is a thinking token, it is catted to the prompt and fed back in. We repeat until we
# get out a normal (non thinking) token. The -crossentropy of this prediction is recorded as the reward
# of this timestep. We then append the *correct* token to the prompt and repeat above until we have N real
# tokens in our context.
#    To train the rl: Once we have produced N real token predictions, we can train our thinking tokens.
# The -losses of each 'timestep' get turned into N reward-to-go scalars values. Using vanilla policy
# gradient, we say the loss at each thinking step is the logprob of the token outputted times the
# accumulated rewards that followed that token. We actually consider every 'chain of thought' (hehe)
# (as in continuous strings of only thinking tokens) to occur in the same timestep, so all in the chain
# take the same reward value.
#    Our total loss is the supervision loss plus some discount factor times the rl loss. this just lets
# us have diff learning rates for the two objectives.

# training in this fashion that I have described basically destroys the niceness of transformer 
# parallelism. shit slow af. without thinking very hard, every modification i can think of that
# improves parallelism is a bit less principled.

# NOTE: I think we need a slightly different reward than just rtg loss. Since we can only have
# 0 loss or positive loss, our reward is only negative or 0. This means thinking tokens will always
# be discouraged every time they are output. (this is fine in normal rl where 'not taking any action'
# is actually not an option so discouraging all actions every time but discouraging some less than 
# others is equivalent to positive reinforcement). Some form of positive reinforcement is needed.
#       A thinktok should have positive reward if it improved supervised prediction accuracy. The heart
# of the issue is that it is quite hard to tell without intractable methods wether a particular token
# was causally useful in later predictions. For example, predicting tokens late in a sequence is easier
# than predictions early in a sequence becuase the model has greater context. And some sequences are 
# just easier to predict than others due to their content. If thninktoks are simply rewarded when the
# later prediction accuracy is high, the model will just learn to output tokens when it thinks the
# prediction task is easy, rather than learning how to output tokens that MAKE the prediction task easier.
#       It seems like the best way to address all these is a valuenet that would be trained, for each
# position in the sequence of real tokens, to estimate the avg future correct token logprob. This would
# be another transformer that should give us a baseline of prediction accuracy so that we can say,
# regardless of the type of sequence or if our prediction is later or earlier in the sequence, wether
# the addition of a certain thinking token was actually better than doing nothing. (with a good valuenet
# it even seems plausible to do search to make better thinktoks than argmaxing)

# or wait why cant we just run the full normal sequence through the model and see its prediction accuracy
# at each token? then directly compare that to the model with thinktoks interspersed and compare token
# to token logit differences?

#      Question: should the valuenet just rate the entire sequence of real tokens, or should it
# evaluate the current context of real + thinking tokens? hmm so the normal rl analogy is weird here.
# What is the valuenet supposed to capture? what should it learn? It's purpose is a way of judging 
# which thinktoks were good and which were bad. what makes thinktoks good or bad? A thinktok's
# goodness is proportional to the amount by which it improves later predictions, and bad by blah blah.
# how do you know if a thinktok improved, harmed, or did not affect later prediction accuracy? You
# can do this by understanding how hard it is to predict the next token given the current context.