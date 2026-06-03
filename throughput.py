import csv, statistics, torch

d = torch.device("cuda:0")
SHAPES = [(5120, 5120), (5120, 13824), (13824, 5120)]
MS = [16, 64, 256, 1024, 4096]
WARM, ITERS = 10, 30

def bench(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def prep_int4(w, gs=128, ikt=8):           # w:[N,K] bf16 -> packed [N,K//2] nibbles
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
    for M in MS:
        b16 = torch.randn(K, N, device=d, dtype=torch.bfloat16)
        a16 = torch.randn(M, K, device=d, dtype=torch.bfloat16)
        flops = 2 * M * N * K; ref = None
        for name in ["bf16", "fp8_e4m3", "int8", "int4"]:
            try:
                if name == "bf16":
                    fn = lambda: torch.matmul(a16, b16)
                elif name == "fp8_e4m3":
                    a8 = a16.to(torch.float8_e4m3fn)
                    b8 = b16.to(torch.float8_e4m3fn).t().contiguous().t()
                    sa = torch.tensor(1.0, device=d); sb = torch.tensor(1.0, device=d)
                    fn = lambda: torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True); fn()
                elif name == "int8":
                    if M <= 16: raise RuntimeError("skip: _int_mm needs M>16")
                    ai = torch.randint(-127, 127, (M, K), device=d, dtype=torch.int8)
                    bi = torch.randint(-127, 127, (N, K), device=d, dtype=torch.int8).t()
                    fn = lambda: torch._int_mm(ai, bi); fn()
                elif name == "int4":
                    packed, saz = prep_int4(b16.t().contiguous())
                    fn = lambda: torch._weight_int4pack_mm(a16, packed, 128, saz); fn()
                ms = bench(fn); tflops = flops / (ms / 1e3) / 1e12
                if name == "bf16": ref = tflops
                sp = tflops / ref if ref else float("nan")
                rows.append(dict(dtype=name, M=M, K=K, N=N, status="ok", ms=round(ms, 4), tflops=round(tflops, 1), speedup=round(sp, 3)))
                print(f"{name:>9} M={M:<5} {K}x{N}  {ms:8.4f}ms  {tflops:8.1f} TFLOP/s  x{sp:5.2f}", flush=True)
            except Exception as ex:
                rows.append(dict(dtype=name, M=M, K=K, N=N, status=f"err:{type(ex).__name__}", ms="", tflops="", speedup=""))
                print(f"{name:>9} M={M:<5} {K}x{N}  {type(ex).__name__}: {str(ex)[:60]}", flush=True)

with open("results/throughput.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote -> results/throughput.csv", flush=True)
