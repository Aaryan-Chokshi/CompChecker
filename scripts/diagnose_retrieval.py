"""Audit actual stored retrieval without loading models or calling Ollama.

Local Qdrant is queried from a SQLite backup to avoid locking a running backend.
Run: python -m scripts.diagnose_retrieval [--document-id UUID --policy-id UUID]
"""

import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from qdrant_client import models

from backend.core.config import Settings
from backend.integrations.vector_store.qdrant import QdrantVectorStore


def read_database(path):
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        documents = {
            row[0]: {"filename": row[1], "role": row[2].lower(), "status": row[3].lower()}
            for row in connection.execute("SELECT id, filename, role, status FROM documents")
        }
        chunks = {
            row[0]: {"document_id": row[1], "text": row[2]}
            for row in connection.execute("SELECT id, document_id, text FROM document_chunks")
        }
        statuses = dict(
            connection.execute("SELECT status, COUNT(*) FROM compliance_runs GROUP BY status")
        )
        latest = connection.execute(
            "SELECT document_id, policy_ids FROM compliance_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return documents, chunks, statuses, latest


def snapshot_local(settings, destination):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "meta.json").write_bytes((settings.qdrant_path / "meta.json").read_bytes())
    source = settings.qdrant_path / "collection" / settings.qdrant_collection / "storage.sqlite"
    target = destination / "collection" / settings.qdrant_collection / "storage.sqlite"
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True) as original:
        with sqlite3.connect(target) as backup:
            original.backup(backup)
    return settings.model_copy(update={"qdrant_path": destination})


def audit(settings, document_id=None, policy_id=None, samples=3):
    documents, chunks, statuses, latest = read_database(settings.database_path)
    if document_id is None or policy_id is None:
        if latest is None:
            raise ValueError("No comparison history. Supply --document-id and --policy-id.")
        document_id = document_id or latest[0]
        policy_id = policy_id or json.loads(latest[1])[0]
    for identifier, role in [(document_id, "subject"), (policy_id, "policy")]:
        if identifier not in documents or documents[identifier]["role"] != role:
            raise ValueError(f"Select an existing {role} document: {identifier}")
    with tempfile.TemporaryDirectory(prefix="compchecker-audit-") as temporary:
        query_settings = (
            settings if settings.qdrant_url else snapshot_local(settings, Path(temporary))
        )
        # Read dimension from the stored collection; this audit does not load embeddings.
        store = QdrantVectorStore(query_settings, lambda: dimension)
        try:
            client = store._get_client()
            config = client.get_collection(settings.qdrant_collection).config.params.vectors
            if not isinstance(config, models.VectorParams):
                raise ValueError("Expected an unnamed-vector collection.")
            dimension = config.size
            points, offset = [], None
            while True:
                batch, offset = client.scroll(
                    collection_name=settings.qdrant_collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                points.extend(batch)
                if offset is None:
                    break
            point_ids = {str(point.id) for point in points}
            mismatches = [
                str(point.id)
                for point in points
                if str(point.id) in chunks
                and (
                    point.payload.get("text") != chunks[str(point.id)]["text"]
                    or point.payload.get("document_id") != chunks[str(point.id)]["document_id"]
                    or point.payload.get("role")
                    != documents[chunks[str(point.id)]["document_id"]]["role"]
                )
            ]
            counts, examples = Counter(), []
            for point in points:
                source_id = point.payload.get("document_id")
                if source_id not in {document_id, policy_id}:
                    continue
                target_id = policy_id if source_id == document_id else document_id
                role = documents[target_id]["role"]
                hits = store.query(
                    point.vector,
                    limit=settings.retrieval_top_k,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=target_id)
                            ),
                            models.FieldCondition(key="role", match=models.MatchValue(value=role)),
                        ]
                    ),
                )
                counts[f"{role}: {len(hits)} hits"] += 1
                if source_id == document_id:
                    examples.append(
                        {
                            "query_chunk_id": str(point.id),
                            "query": point.payload.get("text"),
                            "hits": [
                                {
                                    "chunk_id": hit.id,
                                    "score": hit.score,
                                    "text": hit.payload.get("text"),
                                }
                                for hit in hits
                            ],
                        }
                    )
            examples.sort(
                key=lambda item: item["hits"][0]["score"] if item["hits"] else -1, reverse=True
            )
            return {
                "collection": settings.qdrant_collection,
                "vector_dimension": dimension,
                "sqlite_chunks": len(chunks),
                "qdrant_points": len(points),
                "missing_point_ids": sorted(set(chunks) - point_ids),
                "orphan_point_ids": sorted(point_ids - set(chunks)),
                "payload_mismatches": mismatches,
                "run_status_counts": statuses,
                "subject": {"id": document_id, **documents[document_id]},
                "policy": {"id": policy_id, **documents[policy_id]},
                "query_counts": dict(counts),
                "examples": examples[:samples],
                "note": "Queries reuse indexed vectors. This audits storage, filters and ID mapping, not regenerated embeddings or Ollama decisions.",
            }
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id")
    parser.add_argument("--policy-id")
    parser.add_argument("--samples", type=int, default=3, choices=range(0, 11))
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(Settings(), arguments.document_id, arguments.policy_id, arguments.samples),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
