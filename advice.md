把当前规则改成：

硬拒层：只拦真正危险的证据
补证层：对章节未命中、同义词未覆盖、top5 不够等情况做受控 rescue
覆盖层：补证后仍不满足，才拒答
生成层：仍然只允许 full / guarded_full 进入 LLM

也就是：

旧规则：
partial = 拒答

新规则：
partial → 先判断是否可救回
       → 救回成功 = full / guarded_full
       → 救回失败 = refusal

这样安全性不降，但误杀会少很多。

1. 先重分拒答原因：哪些硬拦，哪些可救回

现在所有原因都被 full-only 压成拒答，太粗。建议先分桶。

A. 继续硬拦的情况

这些是真风险，应该一票否决：

no_relevant_evidence        没有任何相关证据
empty_evidence              证据为空
wrong_source                来源文档不对
version_mismatch            命中过期版本
deleted_or_invisible        文档已删除或不可见
heading_only_after_expand   标题扩展后仍无正文
all_generic_no_body         只有泛化章节，没有正文
low_relevance_after_rescue  补证后仍低相关

这些属于：

证据本身不可用

应该继续拒答。

B. 不要立刻硬拦的情况

这些不一定是证据风险，很多是结构、同义词、排序窗口问题：

section_not_hit
topic_not_hit
partial_term_coverage
generic_only
insufficient_evidence
low_evidence_relevance
heading_only_evidence

它们应该先进入：

coverage_rescue

补证失败再拒答。

2. 修改后的主判定树

建议改成下面这个结构：

召回结果
  ↓
基础证据清洗
  - 排除删除版本
  - 排除错误来源
  - 排除空文本
  - 标题类不算正文证据，但可以作为定位线索
  ↓
如果完全无可用线索
  → refusal

构建 evidence_candidate_pool
  - 不只看 top5
  - 建议看 rerank 后 top10 / top15
  - 只允许同源、正文、当前版本、非泛化 chunk 进入候选
  ↓
第一轮覆盖判定
  ↓
如果 full
  → 允许生成

如果是硬风险
  → refusal

如果是可救回不足
  → coverage_rescue
        - 标题命中则展开子正文
        - 章节未命中则查 section_path / 邻近 heading
        - term 未覆盖则做同义簇归一
        - top5 不够则扩大到 top10/top15
  ↓
补证后重新判定
  ↓
full / guarded_full
  → 允许生成

仍不足
  → refusal

注意：外层仍然可以保持 full-only。

只是内部把“能救回的 partial”提升成 full 或 guarded_full，而不是直接放行所有 partial。

3. 第一层证据有效性要改

你现在第一层有两个容易误杀的点：

只看前 3 个非标题 chunk
总文本长度 < 24 就 insufficient_evidence

这在法规文档里有风险。

有些有效条文很短，比如：

依法给予处罚。
责令限期改正。
可以给予表彰和奖励。

这些字符数不长，但它们是有效证据。

所以 insufficient_evidence 不应该只看长度，应该改成“长度 + 结构信号 + 法规动作词”共同判断。



这样可以避免短条文被误判成 insufficient_evidence。

4. 标题类 chunk 不算证据，但要允许“标题扩展”

现在规则是：

全是标题类 → heading_only_evidence → 拒答

这个太硬。

因为标题命中往往说明找对了位置，只是正文没有被带上来。

应该改成：

标题类 chunk 不算最终证据
但可以触发 heading_expand

也就是：

命中“法律责任”标题
不要直接拒答
而是去找它下面的正文条款
新规则
如果命中的全是标题类：
  1. 找这些 heading 的子 chunk / 后续邻近 chunk
  2. 只取同 source、同 version、同 section_path 下的正文
  3. 找到正文 → 进入 coverage 判定
  4. 找不到正文 → heading_only_after_expand → refusal

这样既不会让标题直接当证据，也不会因为只召回标题而误拒。

5. coverage 不要只看固定 top5

当前规则：

只看前 5 个合格实质证据 chunk

这个对 reranker 非常敏感，也容易误杀。

建议改成两段式：

primary_window = top5
rescue_window = top10 / top15

规则是：

第一轮只看 top5
如果 full，直接通过

如果不是 full：
  在 top10/top15 中找可救回证据

但 rescue 不能无限放大，必须受控。

rescue 候选必须满足
1. 同一个目标文档
2. 当前 active_version
3. 非标题 chunk
4. 非泛化章节，或者能和目标 topic / terms 对上
5. rerank_score / fused_score 不低于救回阈值
6. chunk_role 是正文类
7. 不能来自错误 source


6. section_targets 要分强弱

现在 section_not_hit 很容易误伤，因为有些 query 看起来像章节，其实只是主题。

比如：

奖励与处罚怎么规定？
养犬行为规范有哪些要求？
管理要求是什么？

这些不一定是文档里的精确章节名，更多是用户的主题表达。

所以 section_targets 应该拆成两类：

strong_section_targets
weak_section_or_topic_targets
强章节目标

这些必须严格命中：

第八条
第二章
法律责任
总则
附则
罚则

如果用户明确问：

法律责任一章怎么规定？
第八条怎么说？

那就需要命中对应章节或条号。

否则可以拒答。

弱章节 / 主题目标

这些不应该硬性要求 section_path 命中：

奖励与处罚
管理要求
行为规范
办理流程
登记要求
处罚规定
限制措施

这些允许通过正文语义覆盖救回。

也就是：

弱章节没命中 section_path
但正文覆盖了核心 terms
且来源正确
则可以通过
规则可以这样写
if strong_section_targets:
    if not section_hit(strong_section_targets, evidence):
        return partial_or_refusal("section_not_hit")

if weak_section_targets:
    if section_hit(weak_section_targets, evidence):
        pass
    elif semantic_topic_hit(weak_section_targets, evidence):
        pass
    else:
        return partial_or_refusal("topic_not_hit")

重点是：

强章节查结构
弱主题查语义

不要都按 section_path 死卡。

7. term coverage 要加“法规语义簇”，不是手工文档 alias

你前面担心“每新增一个文档就加映射表”，这个担心对。

不要做文档级 alias 表。

要做的是法规问答通用语义簇。

例如：

LEGAL_TERM_CLUSTERS = {
    "处罚": [
        "处罚", "罚款", "警告", "责令改正", "责令限期改正",
        "没收", "吊销", "追究责任", "法律责任", "依法处理"
    ],
    "奖励": [
        "奖励", "表彰", "鼓励", "扶持", "补助", "资助",
        "给予表彰", "给予奖励"
    ],
    "登记": [
        "登记", "备案", "注册", "申请登记", "办理登记"
    ],
    "管理要求": [
        "应当", "不得", "禁止", "规范", "要求", "义务",
        "责任", "管理", "监督"
    ],
    "限制": [
        "不得", "禁止", "限制", "不得携带", "不得进入",
        "限期", "区域", "时间"
    ],
    "许可": [
        "许可", "审批", "批准", "申请", "核准", "备案"
    ],
}

覆盖判断时不要只做：

处罚 是否字面出现

而要做：

处罚簇 是否被覆盖

例如正文出现：

责令改正
处以罚款
依法追究责任
法律责任

就应该认为“处罚”被覆盖。

8. generic_only 也要改成“泛化不足”而不是直接 partial

现在：

如果前 5 个证据都落在泛化章节里
→ generic_only
→ partial
→ 拒答

这在很多目录/总则/附则场景是合理的。

但也有误杀风险。

比如文档里正文 chunk 的 section_path 缺失，系统可能把它归为无章节，于是误判 generic_only。

建议改成：

generic_only 先检查正文密度
新规则
如果证据都在“总则/附则/目录/无章节”：
  如果没有核心 term 命中
    → generic_only → refusal
  如果有多个核心 term 命中，且 chunk 是正文
    → generic_but_substantive → 允许进入 rescue
  如果能通过邻近 heading 补 section_path
    → 重新判定

也就是说：

泛化章节 + 没有实质内容 = 拒答
泛化章节 + 正文实质内容 = 可救回
9. 建议新增一个内部状态：guarded_full

如果你不想放松 full-only，可以这样设计：

full：天然满足证据门
guarded_full：经过受控 rescue 后满足证据门
partial_unsafe：补证后仍不够
refusal：硬拒

然后生成层允许：

if answer_scope not in {"full", "guarded_full"}:
    refusal

这样比直接允许 partial 安全得多。

guarded_full 的进入条件

必须同时满足：

1. source 正确
2. active_version 正确
3. 至少 1 个正文 chunk
4. 至少 1 个核心 term 或 topic 被覆盖
5. 如果是强章节查询，必须命中章节/条号
6. 不是纯标题
7. 不是纯目录
8. 不是纯泛化章节
9. missing_terms 只能是同义词层面的缺口，不能是核心对象缺失

这样可以救回：

同义词没字面覆盖
section_path 没继承
top5 窗口太窄
标题命中但正文在后面

但不会放过：

错文档
无正文
只有标题
只有目录
泛化套话
证据低相关
10. 推荐的新 evidence gate 伪代码

可以按这个逻辑改：

def evidence_gate(query, ranked_chunks, query_filter):
    targets = extract_targets(query, query_filter)

    candidates = build_candidate_pool(
        ranked_chunks,
        topk=15,
        exclude_deleted=True,
        active_version_only=True,
    )

    hard_result = hard_evidence_check(candidates, targets)

    if hard_result.is_hard_refusal:
        return hard_result

    primary = select_substantive_chunks(
        candidates,
        topk=5,
        min_score=MIN_EVIDENCE_SCORE,
    )

    obs = coverage_check(primary, targets)

    if obs.answer_scope == "full":
        return obs

    if obs.reason in HARD_REFUSAL_REASONS:
        return obs.as_refusal()

    # 可救回原因进入 rescue
    if obs.reason in RESCUABLE_REASONS:
        rescued = coverage_rescue(
            candidates=candidates,
            targets=targets,
            reason=obs.reason,
            max_topk=15,
        )

        obs2 = coverage_check(rescued, targets)

        if obs2.answer_scope == "full":
            obs2.answer_scope = "guarded_full"
            obs2.coverage_reason = "rescued_" + obs.reason
            return obs2

        return obs2.as_refusal("rescue_failed_" + obs.reason)

    return obs.as_refusal()

其中：

HARD_REFUSAL_REASONS = {
    "no_relevant_evidence",
    "empty_evidence",
    "wrong_source",
    "version_mismatch",
    "deleted_or_invisible",
    "heading_only_after_expand",
    "all_generic_no_body",
    "low_relevance_after_rescue",
}

RESCUABLE_REASONS = {
    "heading_only_evidence",
    "insufficient_evidence",
    "low_evidence_relevance",
    "section_not_hit",
    "topic_not_hit",
    "generic_only",
    "partial_term_coverage",
}
11. 每个原原因怎么改
当前原因	建议新处理
no_relevant_evidence	硬拒
empty_evidence	硬拒
heading_only_evidence	先 heading_expand，失败后拒答
insufficient_evidence	检查是否短法规条文，失败再拒答
low_evidence_relevance	在同源 top15 里 rescue，失败再拒答
section_not_hit	强章节硬要求；弱章节转 topic/term 覆盖
topic_not_hit	先做语义簇归一，失败再拒答
generic_only	检查正文密度和 term 覆盖，失败再拒答
partial_term_coverage	做法规语义簇覆盖，失败再拒答
sufficient_evidence	full，通过
12. 最关键的改动点

我会优先改这 5 个地方。

第一，partial_term_coverage 不要直接拒答

改成：

先做 term cluster coverage
再做 evidence rescue
最后仍缺核心 term 才拒答

这是同义词专项集的关键。

第二，section_not_hit 区分强章节和弱主题

改成：

第几条 / 第几章 / 法律责任 / 总则 / 附则
→ 强章节

奖励与处罚 / 管理要求 / 行为规范 / 登记要求
→ 弱主题

强章节没命中可以拒答。

弱主题没命中 section_path，但正文覆盖，可以救回。

第三，标题命中触发正文扩展

不要让：

命中“法律责任”标题

直接变成拒答。

应该尝试找：

法律责任下面的条文

找不到再拒答。

第四，coverage 窗口从 top5 改成 top5 + rescue_top15

最终生成可以只给 top5 证据，但 coverage 判断应该允许从 top15 里补正文证据。

否则 reranker 轻微排序波动就会导致误拒。

第五，证据分数不要只看旧基础相关度

现在 MIN_EVIDENCE_SCORE = 0.6 如果用的是基础相关度，可能和新 reranker 不一致。

建议最终证据分改成组合分：

evidence_score =
  rerank_score
  + source_match_bonus
  + term_hit_bonus
  + section_hit_bonus
  - heading_penalty
  - generic_penalty

不要只靠原始 recall score。

13. 最终推荐配置

可以先保守一点：

MIN_EVIDENCE_SCORE = 0.60
MIN_RESCUE_SCORE = 0.48
MIN_SUBSTANTIVE_CHUNKS = 1

PRIMARY_EVIDENCE_TOPK = 5
RESCUE_EVIDENCE_TOPK = 15

ALLOW_GUARDED_FULL = True
ALLOW_PARTIAL_GENERATION = False

也就是：

partial 仍不生成
guarded_full 才生成

这样不会明显放宽安全边界。

14. 你现在这套规则的最佳修改方向

不要把规则改成“更松”。

应该改成：

更分层
更可救回
更区分风险类型

最终目标是：

错文档、空证据、标题证据、泛化证据、低相关证据
坚决拦下；

正确文档、正确正文、只是同义词没覆盖、章节继承没做好、排序窗口没进 top5
允许受控救回。

我建议你下一步直接按这个顺序改：

1. 新增 HARD_REFUSAL_REASONS / RESCUABLE_REASONS
2. 新增 guarded_full
3. partial_term_coverage 接入法规语义簇
4. section_not_hit 拆强章节 / 弱主题
5. heading_only_evidence 改成 heading_expand
6. coverage 从 top5 改成 top5 + rescue_top15
7. 全量评估按 rescued_reason 分桶

这样修改后，证据门不会变“软”，而是会变“准”：
真正不合格的证据继续挡住，正确证据因为结构和表达差异被误拦的概率会明显下降。