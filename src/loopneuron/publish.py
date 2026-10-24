"""Publish the neuron atlas and study results as a public Hugging Face dataset inside a collection.

Local: huggingface_hub HfApi (https://huggingface.co/docs/huggingface_hub/guides/upload): create the dataset repo,
upload the raw atlas tables (atlas.pt), the token chunks they index (tokens_atlas.npy), the decoded top contexts of
featured neurons (featured_neurons.jsonl), all results JSON, and a dataset card; then add it to a collection
(https://huggingface.co/docs/huggingface_hub/guides/collections).
Global context: last phase `publish`; no model weights are trained or uploaded by this study.
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

COLLECTION_TITLE = "Looped-LM neurons"


def featured_jsonl(atlas, chunks, tok, keys: list[str], path: Path, k: int = 40) -> Path:
    """Decoded top-k contexts (peak marked <<>>) for featured (layer_neuron_pole) keys."""
    from loopneuron import contexts as cx

    with open(path, "w", encoding="utf-8") as f:
        for key in keys:
            layer, neuron, pole = map(int, key.split("_"))
            for s, p, v, loop in cx.neuron_entries(atlas, layer, pole, neuron, k=k):
                f.write(json.dumps({"key": key, "layer": layer, "neuron": neuron, "pole": pole, "chunk": s, "pos": p,
                                    "loop": loop + 1, "activation": v,
                                    "text": cx.window_text(tok, chunks, s, p, mark=True)}) + "\n")
    return path


def publish(repo_id: str, files: dict[str, Path], card: str, log) -> str:
    """Upload `files` ({path_in_repo: local_path}) + README card to a public dataset repo; add it to the collection."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    for dst, src in files.items():
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo_id, repo_type="dataset")
        log.info("uploaded %s -> %s/%s", src, repo_id, dst)
    owner = repo_id.split("/")[0]
    existing = [c for c in api.list_collections(owner=owner) if c.title == COLLECTION_TITLE]
    coll = existing[0] if existing else api.create_collection(COLLECTION_TITLE, private=False,
                                                              description="Neuron-level interpretability of looped LMs")
    try:
        api.add_collection_item(coll.slug, item_id=repo_id, item_type="dataset", exists_ok=True)
    except TypeError:  # older hub versions without exists_ok
        api.add_collection_item(coll.slug, item_id=repo_id, item_type="dataset")
    log.info("collection: https://huggingface.co/collections/%s", coll.slug)
    return coll.slug
