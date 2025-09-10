import sys, pandas as pd
df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]], ignore_index=True)
def p(x): return {"p50": x.quantile(0.5), "p95": x.quantile(0.95)}
grp = df.groupby("concurrency", as_index=False).agg(
    e2e_p50_s=("e2e_s", lambda s: p(s)["p50"]),
    e2e_p95_s=("e2e_s", lambda s: p(s)["p95"]),
    tps=("completion_tokens", lambda s: (df.loc[s.index, "completion_tokens"].sum() / df.loc[s.index, "e2e_s"].sum()))
)
print(grp)
grp.to_csv("./reports/baseline_summary.csv", index=False)
