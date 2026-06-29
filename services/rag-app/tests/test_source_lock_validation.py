from app.core.source.profile import SourceIdentityMixin


class _Common:
    @staticmethod
    def normalize_filename(value):
        return str(value or "").strip()

    @staticmethod
    def normalize_query(value):
        return "".join(str(value or "").split())


class _Runtime:
    common = _Common()

    @staticmethod
    def query_anchor_terms(query):
        terms = ["防火期", "戒严期", "用火", "入山"]
        return [term for term in terms if term in str(query or "")]

    @staticmethod
    def coverage_aspect_variants(value):
        variants = {
            "防火期": ["防火期", "森林防火"],
            "戒严期": ["戒严期", "戒严"],
            "用火": ["用火", "野外用火"],
            "入山": ["入山", "进入林区"],
        }
        return variants.get(value, [value])


class _LexStore:
    def __init__(self, body_by_source):
        self.body_by_source = body_by_source

    def has_section_or_body_like(self, source, pattern):
        needle = str(pattern or "").strip("%")
        return bool(needle and needle in self.body_by_source.get(source, ""))


class _Validator(SourceIdentityMixin):
    def __init__(self, titles, regions, body_by_source):
        self.runtime = _Runtime()
        self._titles = titles
        self._regions = regions
        self.lex_store = _LexStore(body_by_source)

    def display_title(self, source):
        return self._titles.get(source, source)

    def title_alias_candidates(self, source):
        return [self.display_title(source)]

    def source_profile_fields(self, source):
        return {"region": self._regions.get(source, "")}

    def source_state(self, source):
        return {"visible": True}


def test_source_lock_validation_rejects_region_mismatch():
    validator = _Validator(
        titles={"shaoxing.pdf": "绍兴市物业管理条例"},
        regions={"shaoxing.pdf": "绍兴市"},
        body_by_source={"shaoxing.pdf": "业主大会和物业服务管理"},
    )

    result = validator.validate_source_lock_candidate(
        "长春森林防火期和森林防火戒严期分别是什么时间",
        "长春市森林防火条例",
        "shaoxing.pdf",
        prior=0.95,
        match_kind="doc_recall",
    )

    assert result["accepted"] is False
    assert result["hard_negative"] is True
    assert any("region_mismatch" in reason for reason in result["reasons"])


def test_source_lock_validation_accepts_partial_anchor_hits():
    validator = _Validator(
        titles={"forest.docx": "长春市森林资源管理条例"},
        regions={"forest.docx": "长春市"},
        body_by_source={"forest.docx": "森林防火戒严期间，应当加强野外用火和火源管理。"},
    )

    result = validator.validate_source_lock_candidate(
        "长春森林防火期、戒严期和入山规则",
        "长春市森林防火条例",
        "forest.docx",
        prior=0.6,
        match_kind="doc_recall",
    )

    assert result["accepted"] is True
    assert result["anchor_hits"] >= 2
