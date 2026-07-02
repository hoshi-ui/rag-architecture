from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEXT_CACHE_DIR = ROOT / "evals" / "generated" / "uploaded_documents_markdown"
OUT_PATH = ROOT / "evals" / "cases" / "documents_leveled_query_dataset.json"

ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万零〇0-9]+条")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(text: str, limit: int = 520) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    split_at = max(cut.rfind("。"), cut.rfind("；"), cut.rfind("\n"))
    if split_at >= 120:
        return cut[: split_at + 1]
    return cut.rstrip() + "..."


def load_articles(source: str) -> dict[str, str]:
    path = TEXT_CACHE_DIR / f"{source}.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing markdown cache for {source}: {path}")
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    matches = list(ARTICLE_RE.finditer(text))
    articles: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(0)
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = clean_text(text[start:end])
        if len(body) >= len(label) + 8:
            articles.setdefault(label, body)
    return articles


ARTICLE_CACHE: dict[str, dict[str, str]] = {}


def article(source: str, label: str) -> str:
    ARTICLE_CACHE.setdefault(source, load_articles(source))
    try:
        return ARTICLE_CACHE[source][label]
    except KeyError as exc:
        available = ", ".join(list(ARTICLE_CACHE[source])[:12])
        raise KeyError(f"{source} has no {label}; available starts with: {available}") from exc


def evidence(refs: list[tuple[str, str]]) -> list[dict[str, str]]:
    items = []
    for source, label in refs:
        title = source.rsplit("_", 2)[0].removesuffix(".docx").removesuffix(".pdf").removesuffix(".doc")
        items.append(
            {
                "source": source,
                "title": title,
                "clause": label,
                "text": compact(article(source, label), 760),
            }
        )
    return items


def joined_evidence(refs: list[tuple[str, str]]) -> str:
    return "\n\n".join(item["text"] for item in evidence(refs))


def make_case(
    case_id: str,
    *,
    difficulty_category: str,
    difficulty_level: str,
    subtype: str,
    query: str,
    refs: list[tuple[str, str]],
    expected_answer: str | None = None,
    expected_aspects: list[str] | None = None,
    minimum_required_aspect_count: int = 2,
) -> dict[str, Any]:
    expected_sources = list(dict.fromkeys(source for source, _ in refs))
    return {
        "id": case_id,
        "category": {
            "事实检索类": "fact_retrieval",
            "逻辑推理类": "logical_reasoning",
            "多文档综合类": "multi_document_synthesis",
        }[difficulty_category],
        "difficulty_category": difficulty_category,
        "difficulty_level": difficulty_level,
        "subtype": subtype,
        "query": query,
        "expected_behavior": "answer_with_citations",
        "expected_sources": expected_sources,
        "expected_source_policy": "multi_source_required" if len(expected_sources) > 1 else "hard_lock",
        "expected_answer": expected_answer or joined_evidence(refs),
        "expected_evidence": evidence(refs),
        "expected_aspects": expected_aspects or [],
        "must_not_use_sources": [],
        "metric_focus": [
            "source_lock_accuracy",
            "section_recall",
            "citation_support_rate",
            "answer_scope_accuracy",
            "aspect_coverage",
        ],
        "minimum_required_aspect_count": minimum_required_aspect_count,
    }


def make_negative_case(
    case_id: str,
    *,
    subtype: str,
    query: str,
    expected_behavior: str,
    expected_source_policy: str,
    expected_answer: str,
    expected_aspects: list[str],
    must_not_use_sources: list[str] | None = None,
    allowed_retrieval_sources: list[str] | None = None,
    allow_knowledge_base_debunk: bool = False,
    expected_no_retrieval: bool = True,
    expected_signals: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": "negative_control",
        "difficulty_category": "负向控制类",
        "difficulty_level": "negative",
        "subtype": subtype,
        "query": query,
        "expected_behavior": expected_behavior,
        "expected_sources": [],
        "expected_source_policy": expected_source_policy,
        "expected_answer": expected_answer,
        "expected_evidence": [],
        "expected_aspects": expected_aspects,
        "must_not_use_sources": must_not_use_sources or [],
        "allowed_retrieval_sources": allowed_retrieval_sources or [],
        "metric_focus": [
            "negative_control_pass_rate",
            "route_pass_rate",
            "no_retrieval_pass_rate",
            "no_wrong_source_pass_rate",
        ],
        "minimum_required_aspect_count": 2,
        "negative_control": True,
        "expected_signals": expected_signals or [],
        "allow_knowledge_base_debunk": bool(allow_knowledge_base_debunk),
        "expected_no_retrieval": bool(expected_no_retrieval),
    }


S = {
    "linzhi_rental": "林芝市出租房安全管理条例_2024-12-06_2025-01-01.docx",
    "linzhi_legislation": "林芝市地方立法条例_2017-05-26_2017-05-26.pdf",
    "sz_market_docx": "深圳市建筑市场严重违法行为特别处理规定_2007-08-15_.docx",
    "sz_market_pdf": "深圳市建筑市场严重违法行为特别处理规定_2007-08-15_.pdf",
    "sz_waste": "深圳市建筑废弃物减排与利用条例_2009-05-31_.docx",
    "sz_quality": "深圳市建设工程质量管理条例_2004-07-29_2004-07-29.pdf",
    "shaoxing_poetry": "绍兴市浙东唐诗之路文化资源保护和利用条例_2023-04-24_2023-05-01.pdf",
    "shaoxing_property": "绍兴市物业管理条例_2023-06-02_2023-06-02.pdf",
    "liaocheng_dog": "聊城市养犬管理条例_2020-06-15_2020-09-01.docx",
    "changchun_ethnic": "长春市少数民族权益保障条例_2024-04-02_2024-04-02.docx",
    "changchun_cement": "长春市散装水泥管理条例_2018-10-10_2018-11-01.pdf",
    "changchun_forest": "长春市森林资源管理条例_2024-04-02_2024-04-02.docx",
    "changchun_meat": "长春市肉品管理条例__.docx",
    "lingshui_ich": "陵水黎族自治县非物质文化遗产保护条例_2015-04-10_.pdf",
}


def clone_case(base: dict[str, Any], case_id: str, subtype: str, query: str) -> dict[str, Any]:
    item = copy.deepcopy(base)
    item["id"] = case_id
    item["subtype"] = subtype
    item["query"] = query
    return item


def _first_evidence_label(case: dict[str, Any]) -> tuple[str, str]:
    evidence_items = case.get("expected_evidence") if isinstance(case.get("expected_evidence"), list) else []
    if not evidence_items:
        return "", ""
    first = evidence_items[0] if isinstance(evidence_items[0], dict) else {}
    return str(first.get("title") or ""), str(first.get("clause") or "")


def _extend_fact_cases(cases: list[dict[str, Any]]) -> None:
    bases = [case for case in cases if case.get("id", "").startswith("DL-F")]
    templates = [
        ("focused_clause_rewrite", "只根据《{title}》{clause}，回答：{query}"),
        ("citation_required_rewrite", "{query} 请同时给出对应条款依据。"),
        ("key_points_rewrite", "请提炼《{title}》{clause}中与该问题直接相关的规则要点：{query}"),
    ]
    next_index = 11
    for base in bases:
        title, clause = _first_evidence_label(base)
        for subtype_suffix, template in templates:
            case_id = f"DL-F{next_index:03d}"
            query = template.format(title=title, clause=clause, query=base["query"])
            cases.append(clone_case(base, case_id, f"{base['subtype']}_{subtype_suffix}", query))
            next_index += 1


def _extend_logic_cases(cases: list[dict[str, Any]]) -> None:
    bases = [case for case in cases if case.get("id", "").startswith("DL-L")]
    templates = [
        ("condition_first_rewrite", "请先判断适用条件，再回答：{query}"),
        ("conclusion_and_basis_rewrite", "把结论和法规依据分开说明：{query}"),
        ("scenario_reasoning_rewrite", "如果用户描述的事实前提成立，请按条款推导处理结果：{query}"),
    ]
    next_index = 11
    for base in bases:
        for subtype_suffix, template in templates:
            case_id = f"DL-L{next_index:03d}"
            query = template.format(query=base["query"])
            cases.append(clone_case(base, case_id, f"{base['subtype']}_{subtype_suffix}", query))
            next_index += 1


def _extend_multi_cases(cases: list[dict[str, Any]]) -> None:
    bases = [case for case in cases if case.get("id", "").startswith("DL-M")]
    templates = [
        ("grouped_sources_rewrite", "请按来源文件分组作答并比较差异：{query}"),
        ("no_cross_citation_rewrite", "请分别引用每个目标文件，避免把不同文件条款混在一起：{query}"),
    ]
    next_index = 14
    for base in bases:
        for subtype_suffix, template in templates:
            case_id = f"DL-M{next_index:03d}"
            query = template.format(query=base["query"])
            cases.append(clone_case(base, case_id, f"{base['subtype']}_{subtype_suffix}", query))
            next_index += 1

    fee_base = next(case for case in bases if case["id"] == "DL-M013")
    cases.append(
        clone_case(
            fee_base,
            "DL-M040",
            f"{fee_base['subtype']}_matrix_rewrite",
            "请按缴纳主体、收取主体、费用用途、减免条件四个维度，对比聊城养犬管理服务费和长春使用林地费用。",
        )
    )


def _negative_rewrite_prefix(case: dict[str, Any], strict: bool = False) -> str:
    behavior = str(case.get("expected_behavior") or "")
    if strict:
        if behavior == "ask_clarification":
            return "严格禁止用全库检索猜测来源，只判断是否应当澄清："
        if behavior == "evidence_insufficient":
            return "严格拒绝越界法律推断，只依据目标法规判断是否证据不足："
        return "请严格按文件名、地域、格式或版本匹配，不能用相似文件替代："
    if behavior == "ask_clarification":
        return "用户没有给出足够明确的文档来源，判断是否需要澄清："
    if behavior == "evidence_insufficient":
        return "不要引用无关法规，先判断证据是否足以回答："
    return "请处理这个检索请求，找不到精确来源时必须拒绝替代作答："


def _extend_negative_cases(cases: list[dict[str, Any]]) -> None:
    bases = [case for case in cases if case.get("id", "").startswith("DL-N")]
    next_index = 13
    for base in bases:
        case_id = f"DL-N{next_index:03d}"
        query = f"{_negative_rewrite_prefix(base)}{base['query']}"
        cases.append(clone_case(base, case_id, f"{base['subtype']}_guardrail_rewrite", query))
        next_index += 1

    for base in bases[:6]:
        case_id = f"DL-N{next_index:03d}"
        query = f"{_negative_rewrite_prefix(base, strict=True)}{base['query']}"
        cases.append(clone_case(base, case_id, f"{base['subtype']}_strict_validation_rewrite", query))
        next_index += 1


def extend_to_150_cases(cases: list[dict[str, Any]]) -> None:
    _extend_fact_cases(cases)
    _extend_logic_cases(cases)
    _extend_multi_cases(cases)
    _extend_negative_cases(cases)
    if len(cases) != 150:
        raise AssertionError(f"Expected 150 cases after expansion, got {len(cases)}")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate case ids generated")


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    fact_specs = [
        ("DL-F001", "scope", "《林芝市出租房安全管理条例》适用于哪些出租房安全管理活动？", S["linzhi_rental"], "第二条", ["适用地域", "适用活动", "出租房定义", "排除民宿和旅馆业客房"]),
        ("DL-F002", "responsibility", "林芝出租房安全管理中，公安机关、消防救援机构、住房城乡建设部门分别负责什么？", S["linzhi_rental"], "第五条", ["公安机关职责", "消防救援机构职责", "住房城乡建设部门职责"]),
        ("DL-F003", "prohibition", "《聊城市养犬管理条例》对重点管理区饲养烈性犬和大型犬有什么规定？", S["liaocheng_dog"], "第十一条", ["重点管理区禁养", "禁养品种标准制定主体", "导盲犬等例外"]),
        ("DL-F004", "procedure", "聊城市犬只信息登记申请提交后，公安机关应在多久内审查？不符合条件时如何处理？", S["liaocheng_dog"], "第十六条", ["三日内审查", "符合条件发证发牌", "不符合条件说明理由", "十日内处置或送收容场所"]),
        ("DL-F005", "platform", "绍兴物业管理电子信息平台主要用于哪些事项？", S["shaoxing_property"], "第七条", ["交付统计", "业主意见征求和投票", "信用状况公示", "物业服务规范和成本信息", "纠纷投诉"]),
        ("DL-F006", "scope", "《深圳市建筑废弃物减排与利用条例》的适用范围是什么？", S["sz_waste"], "第二条", ["适用于本市行政区域", "建设工程新建改建扩建拆除", "建筑废弃物减排与利用"]),
        ("DL-F007", "penalty_authority", "《深圳市建设工程质量管理条例》第七十九条如何划分行政处罚决定机关？", S["sz_quality"], "第七十九条", ["责令停业整顿等由颁证主管部门决定", "勘察设计质量处罚由规划主管部门决定", "其他处罚由建设或专业工程主管部门决定"]),
        ("DL-F008", "time_period", "长春森林防火期和森林防火戒严期分别是什么时间？", S["changchun_forest"], "第十五条", ["防火期", "戒严期", "防火期野外用火审批", "戒严期禁止一切野外用火"]),
        ("DL-F009", "responsibility", "陵水黎族自治县非物质文化遗产保护工作由哪些主体负责？", S["lingshui_ich"], "第五条", ["自治县人民政府领导", "文化主管部门管理监督", "乡镇和有关部门协助"]),
        ("DL-F010", "scope", "《长春市肉品管理条例》适用于哪些对象和区域？", S["changchun_meat"], "第二条", ["本市行政区域", "肉品管理", "任何单位和个人均须遵守"]),
    ]
    for case_id, subtype, query, source, label, aspects in fact_specs:
        cases.append(
            make_case(
                case_id,
                difficulty_category="事实检索类",
                difficulty_level="easy",
                subtype=subtype,
                query=query,
                refs=[(source, label)],
                expected_aspects=aspects,
                minimum_required_aspect_count=min(3, len(aspects)),
            )
        )

    logic_specs = [
        (
            "DL-L001",
            "deadline_and_authority",
            "出租人在林芝签订房屋租赁合同后第35天仍未备案，应当补办到哪个部门，原本法定期限是多少？",
            [(S["linzhi_rental"], "第十七条")],
            "应指出出租人应自签订房屋租赁合同之日起三十日内，到出租房所在地住房城乡建设部门办理出租房租赁登记备案；第35天未备案已经超过该期限，应补办备案。",
            ["三十日内", "住房城乡建设部门", "超过期限需要补办", "公安机关办理居住登记时有告知义务"],
        ),
        (
            "DL-L002",
            "required_vs_encouraged",
            "林芝集中式租赁住房是否必须安装火灾探测报警器？防烟面罩和救生缓降器属于必须还是鼓励配备？",
            [(S["linzhi_rental"], "第二十二条")],
            "集中式租赁住房出租人应当在每间居室以及公共区域安装火灾探测报警器或者智能火灾预警装置，并在公共区域安装应急照明灯、张贴疏散路线图；防烟面罩、强光手电筒、救生缓降器、自救呼吸器等逃生辅助装置属于鼓励配备。",
            ["集中式租赁住房为应当安装", "公共区域应急照明灯", "每间居室张贴疏散路线图", "逃生辅助装置为鼓励"],
        ),
        (
            "DL-L003",
            "condition_chain",
            "聊城市重点管理区犬只出生满三个月后，免疫、芯片和禁养规则应如何依次判断？",
            [(S["liaocheng_dog"], "第十条"), (S["liaocheng_dog"], "第十一条")],
            "犬只出生满三个月之日起十五日内或者免疫间隔期满前应接受狂犬病免疫并取得免疫证；重点管理区犬只在免疫接种时植入电子芯片；同时重点管理区禁止饲养烈性犬和大型犬，但导盲犬、搜救犬及特殊需要犬只不受该禁养限制。",
            ["出生满三个月起十五日内免疫", "取得狂犬病免疫证", "重点管理区免疫时植入电子芯片", "重点管理区禁养烈性犬和大型犬", "列明例外"],
        ),
        (
            "DL-L004",
            "violation_to_penalty",
            "聊城市养犬人未按规定办理犬只信息登记，逾期仍不登记时，个人和单位分别面临什么后果？",
            [(S["liaocheng_dog"], "第三十八条")],
            "应先责令限期补办；逾期仍不登记的，对个人处二百元以上一千元以下罚款，对单位处一千元以上二千元以下罚款，并可以并处没收犬只。",
            ["责令限期补办", "个人罚款二百元以上一千元以下", "单位罚款一千元以上二千元以下", "可以并处没收犬只"],
        ),
        (
            "DL-L005",
            "date_reasoning",
            "10月20日在长春林区野外用火，是否处于森林防火戒严期？此时入山和用火规则是什么？",
            [(S["changchun_forest"], "第十五条")],
            "10月20日落在9月25日至10月31日的森林防火戒严期内。戒严期内林区禁止一切野外用火，进入林区人员应当办理《入山证》，防火检查人员可检查人员和车辆、扣留不准携带的火种并制止无证人员入山。",
            ["判断10月20日属于戒严期", "戒严期禁止一切野外用火", "进入林区需办理入山证", "防火检查人员权力"],
        ),
        (
            "DL-L006",
            "threshold_reasoning",
            "建设项目在长春需要使用0.5公顷林地时，应由哪一级人民政府审批？如果是1公顷呢？",
            [(S["changchun_forest"], "第二十七条")],
            "使用林地面积0.67公顷以下的，由县级人民政府审批；0.67公顷以上13.34公顷以下的，由市人民政府审批。因此0.5公顷由县级人民政府审批，1公顷由市人民政府审批。",
            ["0.67公顷以下县级审批", "0.67至13.34公顷市级审批", "0.5公顷对应县级", "1公顷对应市级"],
        ),
        (
            "DL-L007",
            "fallback_procedure",
            "绍兴物业项目已具备成立业主大会条件，但建设单位和业主委员会都没有报告，业主可以向谁申请，街道或乡镇应在多久内做什么？",
            [(S["shaoxing_property"], "第十一条")],
            "业主认为物业管理区域已具备成立业主大会法定条件的，可以向物业所在地街道办事处、乡镇人民政府申请成立业主大会；街道办事处、乡镇人民政府应当自收到申请之日起六十日内指导业主成立筹备组，筹备召开首次业主大会会议。",
            ["业主可向街道办事处或乡镇人民政府申请", "六十日内", "指导成立筹备组", "筹备首次业主大会会议"],
        ),
        (
            "DL-L008",
            "rectification_outcome",
            "深圳建筑市场中，工程勘察、设计、施工等单位存在事故隐患，经责令整改后逾期拒不整改，会产生什么处理结果？",
            [(S["sz_market_docx"], "第五条")],
            "工程勘察、设计、施工、监理、检测单位存在事故隐患，经建设行政主管部门责令限期整改，逾期拒不整改或者整改不合格的，由市建设行政主管部门暂扣其资质证书，直至整改合格。",
            ["适用主体", "存在事故隐患", "责令限期整改后逾期拒不整改或整改不合格", "暂扣资质证书直至整改合格"],
        ),
        (
            "DL-L009",
            "approval_exception",
            "深圳政府投资建设的办公场所装修完成后未满八年确需重新装修，是否可以直接装修？需要什么前置条件？",
            [(S["sz_waste"], "第十七条")],
            "不能直接重新装修。政府投资建设的办公场所装修完成后八年内不得重新装修；确需重新装修的，应当经主管部门批准。",
            ["八年内不得重新装修", "确需重新装修", "应经主管部门批准"],
        ),
        (
            "DL-L010",
            "violation_to_penalty",
            "长春屠宰加工厂给畜禽肉品注水且违法所得难以计算时，应如何处罚？情节严重拒不改正会怎样？",
            [(S["changchun_meat"], "第二十条")],
            "肉管办应没收注水或者注入其他物质的肉品和违法所得；违法所得难以计算的，处五百元以上五千元以下罚款。情节严重、拒不改正的，吊销《畜禽屠宰加工许可证》。",
            ["没收注水肉品和违法所得", "违法所得难以计算罚款五百元以上五千元以下", "情节严重拒不改正吊销许可证"],
        ),
    ]
    for case_id, subtype, query, refs, answer, aspects in logic_specs:
        cases.append(
            make_case(
                case_id,
                difficulty_category="逻辑推理类",
                difficulty_level="medium",
                subtype=subtype,
                query=query,
                refs=refs,
                expected_answer=answer,
                expected_aspects=aspects,
                minimum_required_aspect_count=min(4, len(aspects)),
            )
        )

    multi_specs = [
        (
            "DL-M001",
            "scope_compare",
            "比较《林芝市出租房安全管理条例》和《聊城市养犬管理条例》的适用范围：各自覆盖什么管理活动，有哪些排除或特别说明？",
            [(S["linzhi_rental"], "第二条"), (S["liaocheng_dog"], "第二条")],
            "应分别说明：林芝条例适用于本市行政区域内出租房租赁、治安、消防等安全管理及监督活动，并定义出租房且排除民宿、旅馆业客房；聊城条例适用于本市行政区域内养犬行为及相关管理活动，同时说明军用、警用犬不适用，导盲犬、搜救犬和特殊单位犬只有特别规定的依其规定。",
            ["林芝出租房适用范围", "林芝出租房定义和排除", "聊城养犬适用范围", "聊城军警犬排除", "导盲犬等特别说明"],
        ),
        (
            "DL-M002",
            "department_compare",
            "对比林芝出租房安全管理和聊城养犬管理中公安机关的职责差异。",
            [(S["linzhi_rental"], "第五条"), (S["liaocheng_dog"], "第五条")],
            "林芝条例中公安机关负责出租房治安的统一监督管理；聊城条例中公安机关是养犬管理主管部门，负责犬只信息登记、签注、养犬信息系统、犬只收容救助场所管理，以及查处禁养限养、扰民、虐待遗弃、恐吓伤人等违法行为并组织捕捉流浪犬等。答案应体现两者一个偏出租房治安监督，一个偏犬只登记和养犬执法综合管理。",
            ["林芝公安职责", "聊城公安主管部门定位", "聊城登记签注和系统职责", "聊城违法行为查处职责", "差异归纳"],
        ),
        (
            "DL-M003",
            "deadline_compare",
            "比较林芝出租房租赁备案和聊城重点管理区个人养犬登记的办理期限与办理机关。",
            [(S["linzhi_rental"], "第十七条"), (S["liaocheng_dog"], "第十四条")],
            "林芝出租人应自签订房屋租赁合同之日起三十日内，到出租房所在地住房城乡建设部门办理租赁登记备案。聊城重点管理区个人养犬的，养犬人应为完全民事行为能力人，并自取得犬只狂犬病免疫证后五日内，携带犬只到公安机关指定的犬只信息登记场所办理登记。",
            ["林芝三十日内", "林芝住房城乡建设部门", "聊城取得免疫证后五日内", "聊城公安机关指定登记场所", "聊城完全民事行为能力人要求"],
        ),
        (
            "DL-M004",
            "information_system_compare",
            "综合林芝出租房、聊城养犬、绍兴物业三个条例，分别说明其信息平台或电子档案要记录/支撑哪些事项。",
            [(S["linzhi_rental"], "第三十二条"), (S["liaocheng_dog"], "第十九条"), (S["shaoxing_property"], "第七条")],
            "应覆盖三类平台：林芝出租房信息服务与监管平台用于采集、导入、更新出租房相关信息，并共享租赁备案、治安、消防、经营主体登记、公共卫生服务、信用和处罚等信息；聊城养犬信息管理系统生成犬只电子档案，记载养犬人、犬只品种体貌照片、登记证和签注变更注销、免疫、费用缴纳、违法记录等事项；绍兴物业管理电子信息平台用于交付统计、业主意见征求和投票、信用公示、服务规范和成本监测、法律政策咨询、纠纷投诉等。",
            ["林芝平台信息共享范围", "聊城犬只电子档案事项", "绍兴物业平台用途", "三文档均须引用"],
        ),
        (
            "DL-M005",
            "penalty_compare",
            "比较聊城未按规定办理犬只信息登记、长春肉品注水、长春散装水泥现场搅拌三类违法行为的处罚方式和罚款幅度。",
            [(S["liaocheng_dog"], "第三十八条"), (S["changchun_meat"], "第二十条"), (S["changchun_cement"], "第十五条")],
            "应分别说明：聊城未按规定办理犬只信息登记先责令限期补办，逾期仍不登记的个人罚二百元以上一千元以下、单位罚一千元以上二千元以下，并可没收犬只；长春肉品注水由肉管办没收注水肉品和违法所得，违法所得难以计算的罚五百元以上五千元以下，情节严重拒不改正吊销许可证；长春散装水泥现场搅拌混凝土按每立方米一百元、总额不超过五万元罚款，现场搅拌砂浆按每立方米一百元、总额不超过三万元罚款。",
            ["聊城犬只登记处罚", "长春肉品注水处罚", "长春散装水泥现场搅拌处罚", "罚款幅度对比"],
        ),
        (
            "DL-M006",
            "governance_compare",
            "比较林芝出租房、聊城养犬、绍兴物业三个场景中基层组织或社区主体承担的协助职责。",
            [(S["linzhi_rental"], "第六条"), (S["liaocheng_dog"], "第六条"), (S["shaoxing_property"], "第六条")],
            "应指出：林芝乡镇人民政府、街道办事处按属地管理做好出租房安全管理；聊城社区、居（村）民委员会、业主委员会协助宣传教育，可制定养犬公约、划定遛犬区域和时间并监督执行；绍兴居（村）民委员会指导监督业主大会和业主委员会，协助街道办事处、乡镇人民政府做好物业管理，引导业主参与，调解纠纷，协调物业管理与社区建设。",
            ["林芝属地管理", "聊城社区和业委会养犬协助", "绍兴居村委物业协助", "基层治理差异"],
        ),
        (
            "DL-M007",
            "construction_compare",
            "对比深圳建筑市场严重违法行为处理规定、深圳建设工程质量管理条例、深圳建筑废弃物减排与利用条例中与建设单位或工程参与单位相关的义务。",
            [(S["sz_market_docx"], "第八条"), (S["sz_quality"], "第十五条"), (S["sz_waste"], "第十七条")],
            "应分别引用三份深圳文件：建筑市场严重违法行为处理规定要求承接建设工程的勘察、设计、施工、监理、检测、造价咨询、招标代理等单位及注册执业人员向市建设行政主管部门申报信息并建立信用档案；建设工程质量管理条例第十五条应说明其列明的工程质量相关行为要求；建筑废弃物条例要求推行住宅一次性装修，政府投资社会保障性住房应一次性装修，政府投资办公场所装修完成后八年内不得重新装修，确需重新装修需经主管部门批准。",
            ["建筑市场信用档案申报", "建设工程质量义务", "建筑废弃物一次性装修", "三份深圳文件均须引用"],
        ),
        (
            "DL-M008",
            "culture_resource_compare",
            "比较绍兴浙东唐诗之路文化资源保护与陵水非物质文化遗产保护中，政府或主管部门在保护规划方面的职责。",
            [(S["shaoxing_poetry"], "第十条"), (S["lingshui_ich"], "第八条")],
            "应说明绍兴条例第十条中关于浙东唐诗之路文化资源保护和利用相关规划、保护名录或保护措施的要求；陵水条例要求自治县人民政府文化主管部门会同有关部门编制非物质文化遗产保护规划，报自治县人民政府批准后组织实施。答案应体现一个面向区域文化资源保护利用，一个面向非遗保护规划。",
            ["绍兴文化资源保护利用要求", "陵水文化主管部门会同编制规划", "报自治县人民政府批准后实施", "保护对象差异"],
        ),
        (
            "DL-M009",
            "fund_or_fee_compare",
            "比较聊城养犬管理服务费和长春使用林地相关费用：分别由谁收取或缴纳，费用用途或条件是什么？",
            [(S["liaocheng_dog"], "第二十条"), (S["changchun_forest"], "第二十九条")],
            "聊城犬只信息登记、签注应向公安机关交纳养犬管理服务费，收费按有关规定报省有关部门批准后执行，并规定盲人饲养导盲犬、肢体重残人饲养扶助犬免交，凭绝育证明免交。长春经批准使用林地的单位和个人必须支付林地补偿费、林木补偿费、森林植被恢复费和安置补助费，四项费用由县级以上林业行政主管部门收取，并对森林植被恢复费上缴比例及费用用途作出规定。",
            ["聊城养犬服务费缴纳", "聊城免交情形", "长春林地四项费用", "长春收取主体", "费用用途或上缴规则"],
        ),
        (
            "DL-M010",
            "duplicate_source_compare",
            "同名的《深圳市建筑市场严重违法行为特别处理规定》docx 和 pdf 两个文件中，第五条关于事故隐患整改不合格的处理是否一致？请分别引用两个文件。",
            [(S["sz_market_docx"], "第五条"), (S["sz_market_pdf"], "第五条")],
            "应同时引用docx和pdf两个来源，并核对第五条核心内容：工程勘察、设计、施工、监理、检测单位违反安全生产、工程质量管理规定，存在事故隐患，经责令限期整改后逾期拒不整改或者整改不合格的，由市建设行政主管部门暂扣资质证书，直至整改合格。若抽取文本有表格噪声，应说明以可识别条款内容为准。",
            ["docx来源", "pdf来源", "第五条核心处理一致性", "不得只引用其中一个同名文件"],
        ),
        (
            "DL-M011",
            "cross_source_role_comparison_rewrite",
            "分别列出林芝出租房安全管理条例和聊城市养犬管理条例中公安机关的职责，并说明两者职责重点有什么不同，引用不得交叉。",
            [(S["linzhi_rental"], "第五条"), (S["liaocheng_dog"], "第五条")],
            "林芝条例中公安机关负责出租房治安的统一监督管理；聊城条例中公安机关是养犬管理主管部门，负责犬只信息登记、签注、养犬信息系统、犬只收容救助场所管理，以及查处禁养限养、扰民、虐待遗弃、恐吓伤人等违法行为并组织捕捉流浪犬等。答案应体现两者一个偏出租房治安监督，一个偏犬只登记和养犬执法综合管理。",
            ["林芝公安职责", "聊城公安主管部门定位", "聊城登记签注和系统职责", "聊城违法行为查处职责", "差异归纳"],
        ),
        (
            "DL-M012",
            "three_source_grassroots_duties_rewrite",
            "林芝出租房、聊城养犬、绍兴物业三个法规里，基层组织或社区主体分别承担哪些协助、监督或纠纷处理职责？请按文件分组回答。",
            [(S["linzhi_rental"], "第六条"), (S["liaocheng_dog"], "第六条"), (S["shaoxing_property"], "第六条")],
            "应指出：林芝乡镇人民政府、街道办事处按属地管理做好出租房安全管理；聊城社区、居（村）民委员会、业主委员会协助宣传教育，可制定养犬公约、划定遛犬区域和时间并监督执行；绍兴居（村）民委员会指导监督业主大会和业主委员会，协助街道办事处、乡镇人民政府做好物业管理，引导业主参与，调解纠纷，协调物业管理与社区建设。",
            ["林芝属地管理", "聊城社区和业委会养犬协助", "绍兴居村委物业协助", "基层治理差异"],
        ),
        (
            "DL-M013",
            "fee_comparison_rewrite",
            "对比《聊城市养犬管理条例》的养犬管理服务费与《长春市森林资源管理条例》的使用林地费用：缴纳主体、收取主体、费用用途或减免条件分别是什么？",
            [(S["liaocheng_dog"], "第二十条"), (S["changchun_forest"], "第二十九条")],
            "聊城犬只信息登记、签注应向公安机关交纳养犬管理服务费，收费按有关规定报省有关部门批准后执行，并规定盲人饲养导盲犬、肢体重残人饲养扶助犬免交，凭绝育证明免交。长春经批准使用林地的单位和个人必须支付林地补偿费、林木补偿费、森林植被恢复费和安置补助费，四项费用由县级以上林业行政主管部门收取，并对森林植被恢复费上缴比例及费用用途作出规定。",
            ["聊城养犬服务费缴纳", "聊城免交情形", "长春林地四项费用", "长春收取主体", "费用用途或上缴规则"],
        ),
    ]
    for case_id, subtype, query, refs, answer, aspects in multi_specs:
        cases.append(
            make_case(
                case_id,
                difficulty_category="多文档综合类",
                difficulty_level="hard",
                subtype=subtype,
                query=query,
                refs=refs,
                expected_answer=answer,
                expected_aspects=aspects,
                minimum_required_aspect_count=min(5, len(aspects)),
            )
        )

    negative_specs = [
        {
            "case_id": "DL-N001",
            "subtype": "nonexistent_region_title",
            "query": "请回答《不存在的城市出租房安全管理条例》适用于哪些活动？",
            "expected_behavior": "document_not_found",
            "expected_source_policy": "not_found",
            "expected_answer": "应明确找不到指定文档或法规，不应把近似标题的出租房安全管理条例当作答案来源。",
            "expected_aspects": ["识别文档缺失", "返回未找到", "不引用相似来源"],
            "expected_signals": ["document_not_found", "not_found"],
        },
        {
            "case_id": "DL-N002",
            "subtype": "near_title_not_found",
            "query": "《长春市森林防火管理条例》关于森林防火期如何规定？",
            "expected_behavior": "document_not_found",
            "expected_source_policy": "not_found",
            "expected_answer": "应明确找不到指定文档或法规，不应把近似标题的森林资源管理条例当作答案来源。",
            "expected_aspects": ["识别文档缺失", "拒绝近似标题替代", "不引用森林资源管理条例"],
            "must_not_use_sources": [S["changchun_forest"], "长春市森林资源管理条例_2004-08-19_2004-08-19.pdf"],
            "expected_signals": ["document_not_found", "not_found"],
        },
        {
            "case_id": "DL-N003",
            "subtype": "ambiguous_article_without_source",
            "query": "第五条规定了什么？",
            "expected_behavior": "ask_clarification",
            "expected_source_policy": "source_ambiguous",
            "expected_answer": "应要求用户补充具体法规或文件名称，因为仅凭条号无法确定唯一来源。",
            "expected_aspects": ["识别缺少具体文档", "要求补充法规名称", "不进入全库混合检索"],
            "expected_signals": ["document_clarification", "document_ambiguous", "compare_clarification", "ambiguous"],
        },
        {
            "case_id": "DL-N004",
            "subtype": "ambiguous_short_alias",
            "query": "深圳建筑相关条例第五条是什么？",
            "expected_behavior": "ask_clarification",
            "expected_source_policy": "source_ambiguous",
            "expected_answer": "应要求用户明确具体法规文件，因为深圳建筑相关文件存在多份，不能任选其中一份作答。",
            "expected_aspects": ["识别深圳建筑相关文件多义", "要求补充具体法规名称", "不任选单一来源回答"],
            "must_not_use_sources": [S["sz_market_docx"], S["sz_market_pdf"], S["sz_quality"], S["sz_waste"]],
            "expected_signals": ["document_clarification", "document_ambiguous", "compare_clarification", "ambiguous"],
        },
        {
            "case_id": "DL-N005",
            "subtype": "out_of_scope_law",
            "query": "依据《林芝市出租房安全管理条例》，公司裁员补偿应如何计算？",
            "expected_behavior": "evidence_insufficient",
            "expected_source_policy": "evidence_insufficient",
            "expected_answer": "应说明该问题超出出租房安全管理法规的调整范围，当前证据不足以回答劳动补偿问题。",
            "expected_aspects": ["问题超出法规调整范围", "不能引用出租房条例回答劳动问题", "说明证据不足"],
            "allowed_retrieval_sources": [S["linzhi_rental"]],
            "allow_knowledge_base_debunk": True,
            "expected_no_retrieval": False,
            "expected_signals": ["refusal", "evidence_insufficient", "blocked", "out_of_scope"],
        },
        {
            "case_id": "DL-N006",
            "subtype": "missing_duplicate_format",
            "query": "请查询《深圳市建筑市场严重违法行为特别处理规定》xlsx 版本第五条内容",
            "expected_behavior": "document_not_found",
            "expected_source_policy": "not_found",
            "expected_answer": "应明确指定的 xlsx 版本不存在，不应自动回退到同名 docx 或 pdf 文件作答。",
            "expected_aspects": ["指定的 xlsx 文件不存在", "不能回退 docx/pdf 版本", "返回文档未找到或澄清"],
            "must_not_use_sources": [S["sz_market_docx"], S["sz_market_pdf"]],
            "expected_signals": ["document_not_found", "not_found", "document_clarification"],
        },
        {
            "case_id": "DL-N007",
            "subtype": "nonexistent_region_title",
            "query": "请回答《不存在的城市养犬管理条例》关于犬只登记有哪些要求？",
            "expected_behavior": "document_not_found",
            "expected_source_policy": "not_found",
            "expected_answer": "应明确找不到指定城市的养犬管理条例，不应把聊城市养犬管理条例作为替代来源。",
            "expected_aspects": ["识别地域前缀不存在", "返回文档未找到", "不引用聊城市养犬管理条例"],
            "must_not_use_sources": [S["liaocheng_dog"]],
            "expected_signals": ["document_not_found", "not_found"],
        },
        {
            "case_id": "DL-N008",
            "subtype": "business_term_collision",
            "query": "《长春市森林防火条例》第十五条是什么？",
            "expected_behavior": "document_not_found",
            "expected_source_policy": "not_found",
            "expected_answer": "应明确找不到指定的森林防火条例，不能把森林资源管理条例作为近似来源回答。",
            "expected_aspects": ["识别森林防火与森林资源不是同一文件", "返回文档未找到", "不引用长春市森林资源管理条例"],
            "must_not_use_sources": [S["changchun_forest"], "长春市森林资源管理条例_2004-08-19_2004-08-19.pdf"],
            "expected_signals": ["document_not_found", "not_found"],
        },
        {
            "case_id": "DL-N009",
            "subtype": "bare_article_without_source",
            "query": "第二十条规定了什么？",
            "expected_behavior": "ask_clarification",
            "expected_source_policy": "source_ambiguous",
            "expected_answer": "应要求用户补充具体法规或文件名称，因为仅凭第二十条无法确定唯一来源。",
            "expected_aspects": ["识别缺少具体文档", "要求补充法规名称", "不进入全库混合检索"],
            "expected_signals": ["document_clarification", "document_ambiguous", "compare_clarification", "ambiguous"],
        },
        {
            "case_id": "DL-N010",
            "subtype": "missing_duplicate_format",
            "query": "请查询《聊城市养犬管理条例》xlsx 版本第二十条内容",
            "expected_behavior": "document_not_found",
            "expected_source_policy": "not_found",
            "expected_answer": "应明确指定的 xlsx 版本不存在，不应自动回退到同名 docx 文件作答。",
            "expected_aspects": ["指定的 xlsx 文件不存在", "不能回退 docx 版本", "返回文档未找到或澄清"],
            "must_not_use_sources": [S["liaocheng_dog"]],
            "expected_signals": ["document_not_found", "not_found", "document_clarification"],
        },
        {
            "case_id": "DL-N011",
            "subtype": "out_of_scope_law",
            "query": "依据《聊城市养犬管理条例》，员工离职经济补偿怎么计算？",
            "expected_behavior": "evidence_insufficient",
            "expected_source_policy": "evidence_insufficient",
            "expected_answer": "应说明该问题超出养犬管理法规的调整范围，当前证据不足以回答劳动经济补偿问题。",
            "expected_aspects": ["问题超出法规调整范围", "不能引用养犬条例回答劳动问题", "说明证据不足"],
            "allowed_retrieval_sources": [S["liaocheng_dog"]],
            "allow_knowledge_base_debunk": True,
            "expected_no_retrieval": False,
            "expected_signals": ["refusal", "evidence_insufficient", "blocked", "out_of_scope"],
        },
        {
            "case_id": "DL-N012",
            "subtype": "missing_specific_version_year",
            "query": "请查询《长春市森林资源管理条例》2019 年版本第十五条内容",
            "expected_behavior": "document_not_found",
            "expected_source_policy": "not_found",
            "expected_answer": "应明确未找到指定的 2019 年版本，不能回退到 2024 或 2004 年版本作答。",
            "expected_aspects": ["指定年份版本不存在", "不能回退其他年份版本", "返回文档未找到或澄清"],
            "must_not_use_sources": [S["changchun_forest"], "长春市森林资源管理条例_2004-08-19_2004-08-19.pdf"],
            "expected_signals": ["document_not_found", "not_found", "document_clarification"],
        },
    ]
    for spec in negative_specs:
        cases.append(make_negative_case(**spec))

    extend_to_150_cases(cases)

    return cases


def main() -> None:
    cases = build_cases()
    suite = {
        "suite_name": "documents_leveled_query_dataset",
        "schema_version": "0.2",
        "created_for": "基于 documents 目录法规文档构建的分级分类 RAG Query 测试集",
        "source_documents_dir": str(ROOT / "documents"),
        "source_markdown_cache_dir": str(TEXT_CACHE_DIR),
        "description": "将 Query 按难度分为事实检索类、逻辑推理类、多文档综合类；每条包含期望来源、证据条款、期望答案要点与评测关注指标。同时包含负向控制用例，用于验证缺失文档、歧义来源、越界问题和错误格式处理能力。",
        "difficulty_buckets": [
            {"name": "事实检索类", "difficulty_level": "easy", "case_count": 40},
            {"name": "逻辑推理类", "difficulty_level": "medium", "case_count": 40},
            {"name": "多文档综合类", "difficulty_level": "hard", "case_count": 40},
            {"name": "负向控制类", "difficulty_level": "negative", "case_count": 30},
        ],
        "global_metrics": [
            "source_lock_accuracy",
            "section_recall",
            "citation_support_rate",
            "answer_scope_accuracy",
            "aspect_coverage",
            "wrong_source_rate",
            "unsupported_claim_rate",
            "negative_control_pass_rate",
            "route_pass_rate",
            "no_retrieval_pass_rate",
            "no_wrong_source_pass_rate",
        ],
        "case_count": len(cases),
        "cases": cases,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_path": str(OUT_PATH), "case_count": len(cases)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
