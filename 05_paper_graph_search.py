import itertools
import json
import math
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import DATA, FIG, RES, content_runs, load_papers, stem

sns.set_theme(style="white", context="talk")

papers = load_papers()
kw = {p["id"]: p["keywords"] for p in json.loads((DATA / "keywords.json").read_text())}
ids = papers.id.tolist()
title = dict(zip(papers.id, papers.title))
category = dict(zip(papers.id, papers.category))
N = len(ids)

inv = defaultdict(list)                      
for pid, ks in kw.items():
    for k in set(ks):
        inv[k].append(pid)
idf = {k: math.log(N / len(v)) for k, v in inv.items()}

shared = defaultdict(set)                     
for k, ps in inv.items():
    if len(ps) > 60:                           
        continue
    for a, b in itertools.combinations(sorted(ps), 2):
        shared[(a, b)].add(k)

P = nx.Graph()
P.add_nodes_from(ids)
for (a, b), ks in shared.items():
    P.add_edge(a, b, weight=len(ks), idf_weight=sum(idf[k] for k in ks), keywords=sorted(ks))
print(f"Граф публикаций: {P.number_of_nodes()} вершин, {P.number_of_edges()} рёбер")


def short(pid, n=70):
    t = title[pid]
    return t if len(t) <= n else t[:n - 1] + "…"


def similar(pid_or_title, k=5, expand=True):
    """Ближайшие к статье публикации на графе. Сначала по весу прямых рёбер
    (число общих ключевых слов); если соседей меньше k — дополняем
    персонализированным PageRank (пути длины 2+ через общих соседей)."""
    pid = pid_or_title if pid_or_title in P else next(
        (i for i in ids if title[i].lower().startswith(pid_or_title.lower())), None)
    if pid is None:
        raise KeyError(f"статья не найдена: {pid_or_title}")
    direct = sorted(P[pid].items(), key=lambda x: (-x[1]["weight"], -x[1]["idf_weight"]))
    out = [(n, d["weight"], d["keywords"], "прямая связь") for n, d in direct[:k]]
    if expand and len(out) < k:
        ppr = nx.pagerank(P, alpha=0.85, personalization={pid: 1.0}, weight="weight")
        have = {n for n, *_ in out} | {pid}
        for n, s in sorted(ppr.items(), key=lambda x: -x[1]):
            if n in have or s <= 1.0 / N * 1.0001:
                continue
            out.append((n, 0, [], f"через соседей (PPR={s:.4f})"))
            have.add(n)
            if len(out) >= k:
                break
    return pid, out


kw_stems = {k: set(stem(w) for w in k.split()) for k in inv}

_sdf = Counter(s for t in papers.text for s in {stem(w) for w in re.findall(r"[a-z][a-z0-9]+", t.lower())})
stem_idf = lambda s: math.log(N / (1 + _sdf.get(s, 0)))


def query_stems(q: str) -> set:
    s = {s for run in content_runs(q) for s, _ in run}
    return s or {stem(w) for w in re.findall(r"[a-z]+", q.lower())}


def search(q: str, k=10, use_ppr=True, alpha=0.3, ppr_weight="idf_weight"):
    """1) запрос нормализуется так же, как статьи; 2) каждое ключевое слово графа
    получает релевантность = доля его стемов, совпавших с запросом; 3) статья
    получает сумму rel*idf по своим ключевым словам; 4) оценки распространяются
    по графу публикаций персонализированным PageRank, чтобы подтянуть статьи,
    связанные с попавшими в цель, но не содержащие слов запроса."""
    qs = query_stems(q)
    rel = {}
    for w, ks in kw_stems.items():
        hit = qs & ks
        if hit:
            den = sum(max(stem_idf(x), 0.05) for x in ks)
            r = sum(max(stem_idf(x), 0.05) for x in hit) / den
            if r > 0.15:
                rel[w] = r
    direct = defaultdict(float)
    for w, r in rel.items():
        for pid in inv[w]:
            direct[pid] += r * idf[w]
    if not direct:
        return [], {}, rel
    scores = dict(direct)
    if use_ppr:
        tot = sum(direct.values())
        pers = {n: v / tot for n, v in direct.items()}
        ppr = nx.pagerank(P, alpha=0.85, personalization=pers, weight=ppr_weight)
        mx_d, mx_p = max(direct.values()), max(ppr.values())
        scores = {n: (1 - alpha) * direct.get(n, 0) / mx_d + alpha * ppr[n] / mx_p for n in P}
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return ranked, direct, rel



tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = tfidf.fit_transform(papers.text)


def baseline(q, k=10):
    sims = linear_kernel(tfidf.transform([q]), X).ravel()
    return [ids[i] for i in np.argsort(-sims)[:k]]



TESTS = {
    "knowledge graph": r"knowledge graph|\bkgs?\b|graphrag",
    "multimodal image video": r"multimodal|multi-modal|image|visual|video",
    "privacy attacks poisoning": r"privacy|poison|attack|adversarial|security",
    "medical clinical": r"medical|clinical|health|biomedical|patient",
    "recommendation system": r"recommend",
    "agentic agents": r"agent",
    "hallucination faithfulness": r"hallucinat|faithful",
    "multi-hop question answering": r"multi-?hop",
    "code generation software": r"\bcode\b|software|programming",
    "long context window": r"long[- ]context|context window|context length",
}
text_of = dict(zip(papers.id, papers.text.str.lower()))
results = {"queries": {}}
rows = []
for q, pat in TESTS.items():
    rel_set = {i for i in ids if re.search(pat, text_of[i])}
    g, *_ = search(q, 10, use_ppr=True)
    g_no, *_ = search(q, 10, use_ppr=False)
    g10 = [i for i, _ in g]
    g_no10 = [i for i, _ in g_no]
    b10 = baseline(q, 10)
    prec = lambda lst: sum(i in rel_set for i in lst) / max(len(lst), 1)
    rows.append((q, len(rel_set), prec(g_no10), prec(g10), prec(b10)))
    results["queries"][q] = {"relevant_in_corpus": len(rel_set), "P@10_keywords_only": prec(g_no10),
                             "P@10_graph_ppr": prec(g10), "P@10_tfidf_baseline": prec(b10)}
def mean_p10(**kw_):
    return float(np.mean([
        sum(i in {x for x in ids if re.search(pat, text_of[x])} for i, _ in search(q, 10, **kw_)[0]) / 10
        for q, pat in TESTS.items()]))


ablation = {"только ключевые слова": mean_p10(use_ppr=False)}
for wt in ("weight", "idf_weight"):
    for a in (0.2, 0.3, 0.5, 0.7):
        ablation[f"PPR, вес={wt}, alpha={a}"] = mean_p10(use_ppr=True, alpha=a, ppr_weight=wt)
results["ablation_mean_P@10"] = ablation
mean = lambda j: float(np.mean([r[j] for r in rows]))
results["mean_P@10"] = {"keywords_only": mean(2), "graph_ppr": mean(3), "tfidf_baseline": mean(4)}

rng = np.random.default_rng(0)
cat_cnt = Counter(category.values())
same, tot, rand_p = 0, 0, []
for pid in ids:
    if P.degree(pid) == 0:
        continue
    _, out = similar(pid, 5, expand=False)
    same += sum(category[n] == category[pid] for n, *_ in out)
    tot += len(out)
    rand_p.append((cat_cnt[category[pid]] - 1) / (N - 1))
results["similar_same_category_P@5"] = same / tot
results["similar_random_baseline"] = float(np.mean(rand_p))
sim_full = linear_kernel(X)
idx = {p: i for i, p in enumerate(ids)}
g_cos, t_cos, r_cos = [], [], []
for pid in ids[::4]:
    i = idx[pid]
    _, out = similar(pid, 5, expand=False)
    g_cos.append(np.mean([sim_full[i, idx[n]] for n, *_ in out]))
    row = sim_full[i].copy(); row[i] = -1
    t_cos.append(np.sort(row)[-5:].mean())
    r_cos.append(np.mean(rng.choice(np.delete(sim_full[i], i), 5)))
results["similar_cosine"] = {"graph_top5": float(np.mean(g_cos)), "tfidf_top5_upper_bound": float(np.mean(t_cos)),
                             "random": float(np.mean(r_cos))}
results["papers_with_neighbors"] = sum(1 for n in ids if P.degree(n) > 0)
results["graph"] = {"nodes": P.number_of_nodes(), "edges": P.number_of_edges(),
                    "isolated": sum(1 for n in ids if P.degree(n) == 0),
                    "avg_degree": 2 * P.number_of_edges() / N,
                    "max_weight": max(d["weight"] for *_, d in P.edges(data=True)),
                    "components": nx.number_connected_components(P)}
(RES / "search_eval.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

demo = []
demo.append("=== ПОИСК ПОХОЖИХ ПУБЛИКАЦИЙ ===")
heavy = sorted(P.edges(data=True), key=lambda e: -e[2]["weight"])
for pid in [heavy[0][0], papers.id[3], papers.id[100]]:
    src, out = similar(pid, 5)
    demo.append(f"\nИСХОДНАЯ: {title[src]}\n  ключевые слова: {kw[src]}")
    for n, w, ks, how in out:
        demo.append(f"  [{w}] {title[n]}\n        общие: {ks if ks else '—'} ({how})")
demo.append("\n=== ПОИСК ПО ЗАПРОСУ ===")
for q in ["knowledge graph question answering", "multimodal retrieval images", "poisoning attacks defense",
          "hallucination detection", "medical clinical decision support", "recommendation personalized users", "рекомендательные системы"]:
    ranked, direct, rel = search(q, 5)
    if not ranked:
        demo.append(f"\nЗАПРОС: «{q}» — нет совпадений (словарь корпуса англоязычный)")
        continue
    demo.append(f"\nЗАПРОС: «{q}»  (совпавших ключевых слов графа: {len(rel)})")
    for pid, s in ranked:
        tag = "прямое" if pid in direct else "через граф"
        demo.append(f"  {s:.3f} [{tag}] {title[pid]}")
(RES / "search_demo.txt").write_text("\n".join(demo))
print("\n".join(demo))
print(json.dumps(results, ensure_ascii=False, indent=1))


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
degs = [d for _, d in P.degree()]
sns.histplot(degs, bins=40, ax=ax[0], color="#4C72B0"); ax[0].set_yscale("log")
ax[0].set(title="Степень вершин в графе публикаций", xlabel="Соседей", ylabel="Статей (log)")
w = Counter(d["weight"] for *_, d in P.edges(data=True))
ax[1].bar(list(w.keys()), list(w.values()), color="#DD8452"); ax[1].set_yscale("log")
ax[1].set(title="Вес ребра = число общих ключевых слов", xlabel="Общих слов", ylabel="Рёбер (log)")
plt.tight_layout(); plt.savefig(FIG / "11_paper_graph_stats.png", dpi=150); plt.close()

q = "knowledge graph question answering"
ranked, direct, _ = search(q, 12)
top = [p for p, _ in ranked]
S = P.subgraph(top).copy()
fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(S, seed=3, k=1.2, weight="weight")
nx.draw_networkx_edges(S, pos, width=[1 + 1.5 * d["weight"] for *_, d in S.edges(data=True)], alpha=.5, ax=ax)
nx.draw_networkx_nodes(S, pos, node_size=900, ax=ax,
                       node_color=["#55A868" if n in direct else "#C44E52" for n in S], edgecolors="k")
nx.draw_networkx_labels(S, pos, {n: f"{i+1}" for i, n in enumerate(top) if n in S}, font_color="w", ax=ax)
ax.set_title(f"Топ-12 по запросу «{q}» и связи между ними\n(зелёный — прямое совпадение, красный — найдено через граф)")
ax.axis("off")
plt.tight_layout(); plt.savefig(FIG / "12_search_subgraph.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(13, 6.5))
x = np.arange(len(rows)); wd = 0.27
ax.bar(x - wd, [r[2] for r in rows], wd, label="только ключевые слова")
ax.bar(x, [r[3] for r in rows], wd, label="ключевые слова + граф (PPR)")
ax.bar(x + wd, [r[4] for r in rows], wd, label="TF-IDF по тексту (baseline)")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right")
ax.set(ylabel="Precision@10", title="Качество поиска по запросу", ylim=(0, 1.05)); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "13_search_eval.png", dpi=150); plt.close()

for *_, d in P.edges(data=True):
    d["keywords"] = "; ".join(d["keywords"])
nx.write_gexf(P, RES / "paper_graph.gexf")