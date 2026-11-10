"""Task definition: field extraction from a document. One item = (document, requested kind).
Real output: the values, copied from the document, one per line, or NONE. Nothing here is model-specific."""
import ast, hashlib, json, re

MAX_CHARS = 800
PROMPT = ("Extract every {kind} that appears in the document below.\n"
          "Write one value per line, copied exactly from the document, and nothing else. "
          "If there is no {kind} in the document, write NONE.\n\nDocument:\n{doc}")

def kind_text(label: str) -> str:
    return label.replace("_", " ")

def prompt_of(item: dict) -> str:
    return PROMPT.format(kind=kind_text(item["kind"]), doc=item["doc"])

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().strip("\"'`.,;:").strip().casefold()

def parse(output: str) -> set:
    vals = set()
    for ln in str(output).splitlines():
        ln = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", ln)
        v = norm(ln)
        if v:
            vals.add(v)
    if vals == {"none"}:
        return set()
    return vals - {"none"}

def correct(item: dict, output: str) -> bool:
    return parse(output) == {norm(g) for g in item["gold"]}

def sid(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()

def spans_a(row: dict) -> dict:
    """label -> distinct gold values, keeping only values literally present in the document."""
    out, low = {}, row["text"].casefold()
    for s in ast.literal_eval(row["spans"]):
        t = str(s["text"])
        if norm(t) and t.casefold() in low:
            out.setdefault(s["label"], set()).add(t)
    return out

def spans_b(row: dict) -> dict:
    out, low = {}, row["text"].casefold()
    for e in ast.literal_eval(row["entities"]):
        t = str(e["entity"])
        if norm(t) and t.casefold() in low:
            for ty in e.get("types", []):
                out.setdefault(ty, set()).add(t)
    return out

def make_item(src, doc, kind, gold, typ):
    return {"id": sid(src, kind, doc), "src": src, "doc": doc, "kind": kind,
            "gold": sorted(gold), "type": typ}

def pos_items(src, rows, span_fn):
    for r in rows:
        if len(r["text"]) > MAX_CHARS:
            continue
        for k, g in span_fn(r).items():
            yield make_item(src, r["text"], k, g, "pos")

def neg_items(src, rows, span_fn, kinds):
    kinds = sorted(kinds)
    for r in rows:
        if len(r["text"]) > MAX_CHARS:
            continue
        present = span_fn(r)
        h = int(sid("neg", r["text"]), 16)
        for off in range(len(kinds)):
            k = kinds[(h + off) % len(kinds)]
            if k not in present:
                yield make_item(src, r["text"], k, [], "neg")
                break
