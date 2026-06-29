from app.services.embedding import EmbeddingService


def test_normalize_sparse_vector_accepts_mapping_payload():
    assert EmbeddingService.normalize_sparse_vector({"12": 0.5, "42": "1.25", "bad": "x", "7": 0}) == {
        12: 0.5,
        42: 1.25,
    }


def test_parse_embedding_payload_pads_missing_sparse_rows():
    dense, sparse = EmbeddingService._parse_embedding_payload(
        {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "sparse_embeddings": [{"12": 0.5}],
        }
    )

    assert dense == [[0.1, 0.2], [0.3, 0.4]]
    assert sparse == [{12: 0.5}, None]
