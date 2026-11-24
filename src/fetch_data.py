"""
Stream compact subsets of real human vector strokes from Google Quick, Draw!
and inspect flowchart diagram annotations from handwritten-diagram-datasets.
"""
import json
import os
import urllib.request

DATA_DIR = "/home/antigravity/napkin-ml/data"
os.makedirs(DATA_DIR, exist_ok=True)

QUICKDRAW_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
CATEGORIES = ["square", "circle", "diamond", "triangle", "line", "zigzag"]
MAX_SAMPLES_PER_CAT = 3000


def fetch_quickdraw_category(cat: str, limit: int = MAX_SAMPLES_PER_CAT):
    out_path = os.path.join(DATA_DIR, f"qd_{cat}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"[cache] {cat}: {len(existing)} drawings")
        return existing

    url = f"{QUICKDRAW_BASE}/{cat}.ndjson"
    print(f"[fetch] streaming {limit} recognized '{cat}' drawings from Quick, Draw! ...")
    drawings = []
    req = urllib.request.Request(url, headers={"User-Agent": "napkin-ml/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw_line in resp:
            item = json.loads(raw_line.decode("utf-8"))
            if not item.get("recognized", False):
                continue
            strokes = item.get("drawing", [])
            if not strokes or len(strokes) > 4:
                continue
            total_pts = sum(len(s[0]) for s in strokes)
            if total_pts < 4 or total_pts > 60:
                continue
            drawings.append({
                "key_id": item.get("key_id", ""),
                "country": item.get("countrycode", ""),
                "drawing": strokes,
            })
            if len(drawings) >= limit:
                break

    with open(out_path, "w") as f:
        json.dump(drawings, f)
    print(f"[saved] {cat}: {len(drawings)} drawings -> {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
    return drawings


def fetch_fca_coco_stats():
    """Fetch FC_A COCO annotations JSON to compute real handwritten flowchart box stats."""
    out_path = os.path.join(DATA_DIR, "fca_stats.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            stats = json.load(f)
        print(f"[cache] FC_A stats loaded: {stats.get('total_boxes', 0)} boxes")
        return stats

    url = "https://raw.githubusercontent.com/bernhardschaefer/handwritten-diagram-datasets/main/datasets/fca/train.json"
    print("[fetch] downloading FC_A COCO train.json for real flowchart geometry stats...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "napkin-ml/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            coco = json.loads(resp.read().decode("utf-8"))
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        ratios = {}
        for ann in coco.get("annotations", []):
            cname = cats.get(ann["category_id"], "unknown")
            x, y, w, h = ann["bbox"]
            if h > 1 and w > 1:
                ratios.setdefault(cname, []).append(round(w / h, 3))
        summary = {
            "num_images": len(coco.get("images", [])),
            "total_boxes": len(coco.get("annotations", [])),
            "categories": {
                k: {
                    "count": len(v),
                    "median_aspect": round(sorted(v)[len(v) // 2], 3),
                }
                for k, v in ratios.items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] FC_A stats: {summary['num_images']} diagrams, {summary['total_boxes']} annotations")
        return summary
    except Exception as e:
        print(f"[warn] Could not fetch FC_A stats ({e}); continuing with Quick, Draw! vector data.")
        return {}


if __name__ == "__main__":
    for c in CATEGORIES:
        fetch_quickdraw_category(c)
    fetch_fca_coco_stats()
