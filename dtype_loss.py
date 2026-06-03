import argparse, csv, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def quant_int(w, bits, group=128):
    out, inn = w.shape
    g = group if inn % group == 0 else inn
    wg = w.float().reshape(out, inn // g, g)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    qmax = 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(wg / (amax / qmax)), -qmax - 1, qmax) * (amax / qmax)
    return q.reshape(out, inn).to(w.dtype)

def quant_fp8(w):
    wf = w.float(); amax = wf.abs().amax(1, keepdim=True).clamp_min(1e-8)
    scale = 448.0 / amax
    return ((wf * scale).to(torch.float8_e4m3fn).float() / scale).to(w.dtype)

def _fake_float(x, e, m):
    sign = torch.sign(x); ax = x.abs().clamp_min(1e-30)
    bias = 2 ** (e - 1) - 1
    ex = torch.floor(torch.log2(ax)).clamp(min=1 - bias, max=bias)
    step = torch.pow(2.0, ex - m); q = torch.round(ax / step) * step
    q = q.clamp(max=(2 - 2 ** (-m)) * (2.0 ** bias))
    return torch.where(x == 0, torch.zeros_like(x), sign * q)

def quant_float(w, e, m):                       # scaled emulation (fixes underflow)
    maxv = (2 - 2 ** (-m)) * (2.0 ** (2 ** (e - 1) - 1))
    wf = w.float(); amax = wf.abs().amax(1, keepdim=True).clamp_min(1e-8)
    scale = maxv / amax
    return (_fake_float(wf * scale, e, m) / scale).to(w.dtype)

def cast_rt(dt): return lambda w: w.to(dt).to(w.dtype)

DTYPES = [
    ("fp128", 128, "unavailable", None),
    ("fp64", 64, "native", cast_rt(torch.float64)),
    ("fp32", 32, "native", cast_rt(torch.float32)),
    ("fp16", 16, "native(ref)", lambda w: w.clone()),
    ("bf16", 16, "native", cast_rt(torch.bfloat16)),
    ("fp8_e4m3", 8, "native-h200", quant_fp8),
    ("fp6_e3m2", 6, "emulated", lambda w: quant_float(w, 3, 2)),
    ("int8", 8, "RTN/channel", lambda w: quant_int(w, 8, 10 ** 9)),
    ("int4", 4, "RTN/group128", lambda w: quant_int(w, 4, 128)),
    ("int2", 2, "RTN/group128", lambda w: quant_int(w, 2, 128)),
]

@torch.no_grad()
def ppl(model, ids):
    logits = model(ids).logits[:, :-1].float(); tgt = ids[:, 1:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    return ce.item(), math.exp(ce.item())

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="./llama2-13b-fp16")
ap.add_argument("--out", default="results/dtype_loss.csv")
ap.add_argument("--seq", type=int, default=4096)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="auto").eval()
in_dev = next(model.parameters()).device

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(t for t in ds["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[:, :a.seq].to(in_dev)
print(f"eval tokens: {ids.shape[1]} (wikitext-2 test)", flush=True)

linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
masters = {id(m): m.weight.data.clone() for m in linears}
print(f"quantizing {len(linears)} Linear layers per dtype", flush=True)

_, ref = ppl(model, ids)                        # fp16 native reference, computed first
print(f"reference fp16 ppl = {ref:.4f}\n", flush=True)

fp8_ok = hasattr(torch, "float8_e4m3fn"); rows = []
for name, bits, kind, fn in DTYPES:
    if fn is None or (name.startswith("fp8") and not fp8_ok):
        rows.append(dict(dtype=name, bits=bits, kind=kind, status="skipped", ce="", ppl="", ppl_ratio=""))
        print(f"{name:>9} {bits:>4}b {kind:<13} skipped", flush=True); continue
    for m in linears: m.weight.data = fn(masters[id(m)]).to(m.weight.device)
    ce, p = ppl(model, ids); r = p / ref
    rows.append(dict(dtype=name, bits=bits, kind=kind, status="ok", ce=round(ce, 5), ppl=round(p, 4), ppl_ratio=round(r, 4)))
    print(f"{name:>9} {bits:>4}b {kind:<13} ce={ce:7.4f} ppl={p:11.4f} x{r:9.4f}", flush=True)
for m in linears: m.weight.data = masters[id(m)]

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nwrote -> {a.out}", flush=True)
