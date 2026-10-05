"""Dump key metrics from mainrun logs. Usage: python3 analyze.py <glob-or-files...>"""
import json, sys, glob, statistics

files = []
for a in sys.argv[1:]:
    files += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]

rows = []
for f in files:
    try:
        L = [json.loads(x) for x in open(f)]
    except Exception as e:
        print(f"{f}: unreadable ({e})"); continue
    ev = {}
    for x in L:
        ev.setdefault(x["event"], x)
    hp = ev.get("hyperparameters_configured", {})
    di = ev.get("dataset_info", {})
    mi = ev.get("model_info", {})
    vs = [x for x in L if x["event"] == "validation_step"]
    ts = ev.get("training_summary", {})
    tr = [x["loss"] for x in L if x["event"] == "training_step"]
    ov = ev.get("tokenizer_overlap_info")
    if not vs:
        print(f"{f}: no validation_step (crashed?) — last line: {L[-1] if L else 'empty'}")
        continue
    rows.append(dict(
        f=f.split("/")[-1], label=hp.get("run_label", ""), opt=hp.get("optimizer", "?"),
        block=hp.get("block_size"), batch=hp.get("batch_size"), drop=hp.get("dropout"),
        vocab=hp.get("vocab_size"), nl=hp.get("n_layer"), dm=hp.get("d_model"),
        sri=hp.get("scaled_residual_init"), shuf=hp.get("shuffle_batches"),
        tokscope=hp.get("tokenizer_scope"),
        max_steps=di.get("max_steps"), avgtok=di.get("avg_tokens_per_val_title"),
        params=mi.get("parameters_count"),
        final=round(vs[-1]["loss"], 5), best=round(ts.get("best_val_loss", vs[-1]["loss"]), 5),
        final_tl=round(vs[-1].get("train_loss") or 0, 3),
        early_spk=round(statistics.pstdev(tr[:200]), 4) if len(tr) >= 5 else None,
        overlap=(round(ov["vocab_overlap_frac"], 4) if ov else None),
        tail=[round(x["loss"], 4) for x in vs[-3:]],
    ))

for r in sorted(rows, key=lambda r: r["final"]):
    print("%-34s opt=%-5s b=%-3s bt=%-4s dr=%-4s sri=%-5s shuf=%-5s tok=%-13s "
          "ms=%-5s p=%-9s fin=%.5f best=%.5f tl=%.2f spk=%s ov=%s tail=%s" % (
        r["label"] or r["f"], r["opt"], r["block"], r["batch"], r["drop"], r["sri"], r["shuf"],
        r["tokscope"], r["max_steps"], r["params"], r["final"], r["best"], r["final_tl"],
        r["early_spk"], r["overlap"], r["tail"]))
