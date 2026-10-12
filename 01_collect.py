import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
QUERY = ('(ti:"retrieval-augmented generation" OR abs:"retrieval-augmented generation" '
         'OR abs:"retrieval augmented generation") '
         'AND submittedDate:[202001010000 TO 202612312359]')
TARGET = 800
PAGE = 100
OUT = Path(__file__).parent / "data" / "papers.csv"


def fetch(start: int) -> list[dict]:
    params = {"search_query": QUERY, "start": start, "max_results": PAGE,
              "sortBy": "relevance", "sortOrder": "descending"}
    for attempt in range(5):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 200:
            break
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    root = ET.fromstring(r.content)
    rows = []
    for e in root.findall("a:entry", NS):
        rows.append({
            "id": e.find("a:id", NS).text.strip(),
            "title": " ".join(e.find("a:title", NS).text.split()),
            "authors": "; ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS)),
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
            "published": e.find("a:published", NS).text[:10],
            "category": e.find("{http://arxiv.org/schemas/atom}primary_category").get("term"),
        })
    return rows


def main():
    OUT.parent.mkdir(exist_ok=True)
    rows = []
    for start in range(0, TARGET, PAGE):
        batch = fetch(start)
        print(f"start={start}: получено {len(batch)}")
        rows += batch
        if len(batch) < PAGE:
            break
        time.sleep(3)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df.to_csv(OUT, index=False)
    print(f"Сохранено {len(df)} публикаций, {df.published.min()} .. {df.published.max()} -> {OUT}")


if __name__ == "__main__":
    main()
