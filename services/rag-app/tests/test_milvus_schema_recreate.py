from app.storage.milvus import VectorDBService


class _DataType:
    ARRAY = "ARRAY"


class _Client:
    def __init__(self, field_type):
        self.field_type = field_type

    def describe_collection(self, collection_name):
        return {"fields": [{"name": "applicable_subjects", "type": self.field_type}]}


def test_legacy_json_applicable_subjects_schema_requires_recreate():
    db = VectorDBService()
    db.client = _Client("JSON")

    assert db._collection_has_legacy_applicable_subjects_schema(_DataType)


def test_array_applicable_subjects_schema_does_not_require_recreate():
    db = VectorDBService()
    db.client = _Client("ARRAY")

    assert not db._collection_has_legacy_applicable_subjects_schema(_DataType)
