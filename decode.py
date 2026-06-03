import csv, statistics, torch
d = torch.device("cuda:0")
SHAPES = [(5120, 5120), (5120, 13824), (13824, 5120)]
MS = [1, 2, 4, 8]
WARM, ITERS = 20, 50

def bench(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def prep_int4(w, gs=128, ikt=8):
    N, K = w.shape
    tq = w.reshape(-1, gs).float()
    mn = tq.amin(1, keepdim=True); mx = tq.amax(1, keepdim=True)
    scales = (mx - mn).clamp(min=1e-6) / 15.0; zeros = mn + scales * 8
    q = tq.sub(mn).div(scales).round().clamp(0, 15).to(torch.uint8).reshape(N, K)
    qp = (q[:, ::2] | (q[:, 1::2] << 4)).contiguous()
    packed = torch._convert_weight_to_int4pack(qp, ikt)
    saz = torch.stack([scales, zeros], -1).reshape(N, K // gs, 2).transpose(0, 1).contiguous().to(torch.bfloat16)
    return packed, saz

rows = []
for K, N in SHAPES:
    b16 = torch.randn(K, N, device=d, dtype=torch.bfloat16)
    packed, saz = prep_int4(b16.t().contiguous())
    for M in MS:
        a16 = torch.randn(M, K, device=d, dtype=torch.bfloat16)
        t_bf16 = bench(lambda: torch.matmul(a16, b16))
        t_int4 = bench(lambda: torch._weight_int4pack_mm(a16, packed, 128, saz))
        sp = t_bf16 / t_int4
        rows.append(dict(M=M, K=K, N=N, bf16_ms=round(t_bf16, 5), int4_ms=round(t_int4, 5), int4_latency_speedup=round(sp, 3)))
        print(f"M={M:<3} {K}x{N}  bf16 {t_bf16:.5f}ms  int4 {t_int4:.5f}ms  int4 speedup x{sp:.2f}", flush=True)

with open("results/decode_int4.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote -> results/decode_int4.csv", flush=True)
