import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import FIG, RES, content_runs, load_papers, tokens

sns.set_theme(style="whitegrid", context="talk")
df = load_papers()
out = {}

df["n_authors"] = df.authors.str.count(";") + 1
df["abs_words"] = df.abstract.map(lambda s: len(tokens(s)))
df["title_words"] = df.title.map(lambda s: len(tokens(s)))
out["n_papers"] = len(df)
out["date_min"], out["date_max"] = df.published.min(), df.published.max()
out["abs_words_mean"] = float(df.abs_words.mean())
out["abs_words_median"] = float(df.abs_words.median())
out["title_words_mean"] = float(df.title_words.mean())
out["authors_mean"] = float(df.n_authors.mean())
out["by_year"] = df.year.value_counts().sort_index().to_dict()
out["top_categories"] = df.category.value_counts().head(10).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
df.year.value_counts().sort_index().plot.bar(ax=ax[0], color="#4C72B0")
ax[0].set(title="Публикации по годам", xlabel="Год", ylabel="Число статей")
q = df.published.str[:7].value_counts().sort_index()
q = q[q.index >= "2023-01"]
q.plot(ax=ax[1], marker="o", color="#DD8452")
ax[1].set(title="Публикации по месяцам (с 2023)", xlabel="Месяц", ylabel="Число статей")
ax[1].tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(FIG / "01_dynamics.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
df.category.value_counts().head(10).sort_values().plot.barh(ax=ax[0], color="#55A868")
ax[0].set(title="Топ-10 категорий arXiv", xlabel="Статей")
sns.histplot(df.abs_words, bins=30, ax=ax[1], color="#C44E52")
ax[1].axvline(df.abs_words.median(), color="k", ls="--")
ax[1].set(title="Длина аннотации (слов)", xlabel="Слов", ylabel="Статей")
sns.histplot(df.n_authors.clip(upper=20), bins=20, ax=ax[2], color="#8172B2")
ax[2].set(title="Авторов на статью (обрезано на 20)", xlabel="Авторов", ylabel="Статей")
plt.tight_layout(); plt.savefig(FIG / "02_categories_lengths.png", dpi=150); plt.close()

authors = Counter(a for s in df.authors for a in s.split("; "))
out["n_unique_authors"] = len(authors)
out["top_authors"] = authors.most_common(15)
top = authors.most_common(15)[::-1]
plt.figure(figsize=(10, 8))
plt.barh([a for a, _ in top], [c for _, c in top], color="#64B5CD")
plt.title("Самые продуктивные авторы"); plt.xlabel("Статей")
plt.tight_layout(); plt.savefig(FIG / "03_authors.png", dpi=150); plt.close()

raw = Counter(w for t in df.text for w in tokens(t))
content = Counter(s for t in df.text for run in content_runs(t) for s, _ in run)
bigrams = Counter()
for t in df.text:
    for run in content_runs(t):
        stems = [s for s, _ in run]
        bigrams.update(zip(stems, stems[1:]))


def zipf_fit(counter, kmin=1, kmax=None):
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    sl = slice(kmin - 1, kmax)
    slope, intercept = np.polyfit(np.log10(ranks[sl]), np.log10(freqs[sl]), 1)
    return ranks, freqs, slope, intercept


fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
for a, (name, c, kmax) in zip(ax, [("Все слова (сырой текст)", raw, 1000),
                                   ("Содержательные слова (стемы)", content, 1000),
                                   ("Биграммы содержательных слов", bigrams, 1000)]):
    ranks, freqs, slope, ic = zipf_fit(c, 1, kmax)
    a.loglog(ranks, freqs, ".", alpha=.6, label="эмпирика")
    a.loglog(ranks, 10 ** ic * ranks ** slope, "r-", lw=2, label=f"fit: наклон = {slope:.2f}")
    a.loglog(ranks, freqs[0] * ranks ** -1.0, "k--", lw=1, label="идеал Ципфа (наклон −1)")
    a.set(title=name, xlabel="Ранг", ylabel="Частота"); a.legend()
    key = {"Все слова (сырой текст)": "zipf_raw", "Содержательные слова (стемы)": "zipf_content",
           "Биграммы содержательных слов": "zipf_bigrams"}[name]
    out[key] = {"slope": float(slope), "vocab": len(c), "tokens": int(sum(c.values()))}
plt.tight_layout(); plt.savefig(FIG / "04_zipf.png", dpi=150); plt.close()

rng = np.random.default_rng(0)
order = rng.permutation(len(df))
seen, vocab_curve, tok_curve, n = set(), [], [], 0
for i in order:
    for w in tokens(df.text.iloc[i]):
        n += 1; seen.add(w)
    tok_curve.append(n); vocab_curve.append(len(seen))
k, logb = np.polyfit(np.log10(tok_curve), np.log10(vocab_curve), 1)
out["heaps_beta"] = float(k); out["heaps_K"] = float(10 ** logb)
plt.figure(figsize=(8, 6))
plt.loglog(tok_curve, vocab_curve, label="эмпирика")
plt.loglog(tok_curve, 10 ** logb * np.array(tok_curve) ** k, "r--", label=f"K·N^β, β={k:.2f}")
plt.xlabel("Токенов N"); plt.ylabel("Словарь V"); plt.title("Закон Хипса"); plt.legend()
plt.tight_layout(); plt.savefig(FIG / "05_heaps.png", dpi=150); plt.close()

from common import STOP
top_words = Counter({w: c for w, c in raw.items() if w not in STOP and len(w) > 2}).most_common(25)
out["top_words"] = top_words
plt.figure(figsize=(10, 9))
plt.barh([w for w, _ in top_words][::-1], [c for _, c in top_words][::-1], color="#937860")
plt.title("Топ-25 слов (без стоп-слов)"); plt.xlabel("Частота")
plt.tight_layout(); plt.savefig(FIG / "06_top_words.png", dpi=150); plt.close()

(RES / "eda.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=int))
print(json.dumps(out, ensure_ascii=False, indent=2, default=int))
