"""Общие утилиты: загрузка корпуса, токенизация, нормализация, извлечение n-грамм."""
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
RES = ROOT / "results"
for d in (DATA, FIG, RES):
    d.mkdir(exist_ok=True)

# слова, которые есть в каждой статье и не несут тематической информации
BOILERPLATE = set("""
paper papers study studies work works approach approaches method methods propose proposed
present presents introduce introduces show shows shown demonstrate demonstrates result results
using use used uses based novel new existing recent recently however also can may across
within without among via thus including include includes order significantly significant
experiments experimental experiment evaluate evaluation evaluated evaluating performance
improve improves improved improvement improving effective effectiveness different various
address addresses challenge challenges problem problems task tasks case cases large
real world data dataset datasets benchmark benchmarks achieve achieves achieved state art
framework system systems provide provides provided enable enables enabled well high
number time need needs able make makes like often one two three first second
end instance instances objective consistent improvements baseline baselines systematically
investigate investigates fundamental impact enhancing enhance enhances relevant conventional
looking employing employ employs samples sample stage wise key main major simple strong
extensive comprehensive substantial notable remarkable consistently outperform outperforms
outperforming outperformed superior compared compare compares demonstrating demonstrated
leverage leverages leveraging incorporate incorporates incorporating integrate integrates
integrating particularly specifically especially despite although while whether suffer suffers
limited limitation limitations lack lacks leading leads lead effectively efficiently
advance advances advanced develop develops developed design designed designs apply applying
applied application applications aim aims aimed focus focuses approach potential
findings underscore underscores wide range percentage points remains underexplored combines
highlight highlights highlighting reveal reveals revealed combine combining offers offer
significantly gains gain trade offs additional additionally further furthermore moreover
widely commonly increasingly rapidly demonstrates establishes establish
""".split())
STOP = set(ENGLISH_STOP_WORDS) | BOILERPLATE
# слова самого запроса: присутствуют в каждой публикации
QUERY_WORDS = {"retrieval", "augmented", "generation", "rag"}

_stem = SnowballStemmer("english").stem
_TOKEN = re.compile(r"[a-z][a-z0-9]+")
_PHRASE = re.compile(r"retrieval[\s-]+augmented(?:[\s-]+generation)?|\brags?\b|\bretrieval[\s-]+enhanced[\s-]+generation")
_MULTI = re.compile(r"\bmulti[\s-]+(?=[a-z])")
_CHUNK = re.compile(r"[.,;:!?()\[\]{}\"“”/|]+")


def load_papers() -> pd.DataFrame:
    df = pd.read_csv(DATA / "papers.csv")
    df["text"] = df["title"] + ". " + df["abstract"]
    df["year"] = df["published"].str[:4].astype(int)
    return df


def tokens(text: str) -> list[str]:
    """Нижний регистр, дефисы -> пробелы, только слова из латиницы."""
    return _TOKEN.findall(text.lower().replace("-", " "))


def stem(w: str) -> str:
    return _stem(w)


def content_runs(text: str) -> list[list[tuple[str, str]]]:
    """Список непрерывных фрагментов из содержательных слов: [(stem, surface), ...].
    Знаки препинания и стоп-слова разрывают фрагмент, поэтому n-граммы
    не пересекают границы фраз."""
    runs = []
    text = _PHRASE.sub(" , ", text.lower())
    text = _MULTI.sub("multi", text).replace("-", " ")
    for chunk in _CHUNK.split(text):
        run = []
        for w in _TOKEN.findall(chunk):
            if w in STOP or len(w) < 3 or w.isdigit():
                if run:
                    runs.append(run)
                run = []
            else:
                run.append((stem(w), w))
        if run:
            runs.append(run)
    return runs


def ngrams_of(text: str, nmax: int = 3):
    """Возвращает Counter {tuple(stems): count} и Counter поверхностных форм."""
    counts, surf = Counter(), {}
    for run in content_runs(text):
        for n in range(1, nmax + 1):
            for i in range(len(run) - n + 1):
                gram = run[i:i + n]
                key = tuple(s for s, _ in gram)
                if n == 1 and key[0] in {stem(w) for w in QUERY_WORDS}:
                    continue
                counts[key] += 1
                surf.setdefault(key, Counter())[" ".join(w for _, w in gram)] += 1
    return counts, surf
