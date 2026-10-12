import json
import math
from collections import Counter

from common import DATA, RES, load_papers, ngrams_of

MIN_DF = 4          
MAX_DF_RATIO = 0.35  
TOP_K = 12          
TITLE_BOOST = 2     

df = load_papers()
N = len(df)

per_doc, surface = [], {}
for _, r in df.iterrows():
    c, s = ngrams_of(r.abstract)
    t, st = ngrams_of(r.title)
    for k, v in t.items():
        c[k] += v * (TITLE_BOOST - 1)
    for src in (s, st):
        for k, cnt in src.items():
            surface.setdefault(k, Counter()).update(cnt)
    per_doc.append(c)

dfreq = Counter(k for c in per_doc for k in c)
vocab = {k for k, d in dfreq.items() if MIN_DF <= d <= MAX_DF_RATIO * N}
print(f"кандидатов всего: {len(dfreq)}, после фильтра по df: {len(vocab)}")


def label(key):
    return surface[key].most_common(1)[0][0]


def contained(a, b):
    """a — подпоследовательность b (подряд идущие стемы)"""
    return len(a) < len(b) and any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))


keywords = []
for c in per_doc:
    scored = []
    for k, tf in c.items():
        if k not in vocab:
            continue
        idf = math.log(N / dfreq[k])
        scored.append((tf * idf * (1 + 0.25 * (len(k) - 1)), k))
    scored.sort(reverse=True)
    cand = [k for _, k in scored[:TOP_K * 2]]
    kept = [k for k in cand if not any(contained(k, o) for o in cand)]
    keywords.append([label(k) for k in kept[:TOP_K]])

df["keywords"] = keywords
(DATA / "keywords.json").write_text(json.dumps(
    [{"id": i, "title": t, "keywords": k} for i, t, k in zip(df.id, df.title, keywords)],
    ensure_ascii=False, indent=1))

freq = Counter(k for ks in keywords for k in ks)
n_by_len = Counter(len(k.split()) for k in freq)
stats = {"unique_keywords": len(freq), "by_ngram_len": dict(n_by_len),
         "avg_per_paper": sum(map(len, keywords)) / N, "top30": freq.most_common(30)}
(RES / "keywords_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print(json.dumps(stats, ensure_ascii=False, indent=1))
for i in range(5):
    print("\n", df.title.iloc[i], "\n  ->", keywords[i])
