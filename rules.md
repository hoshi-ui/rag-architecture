一、上传入口校验规则要补强
你现在只校验：
文件名不能为空
文件内容不能为空
这太薄了。建议新增以下校验。
1. 文件大小上限

至少要有：

单文件最大大小
批量上传最大总大小
PDF 最大页数
图片最大像素尺寸
Excel 最大 sheet / 行 / 列数量

否则大 PDF、大图、异常 Excel 很容易拖死 OCR、解析和 embedding。

建议规则：

{
  "max_file_size_mb": 100,
  "max_pdf_pages": 300,
  "max_image_pixels": 40000000,
  "max_xlsx_rows": 20000,
  "max_xlsx_sheets": 50
}

如果你后面做企业知识库，限制可以配置化，不要写死。

2. MIME / magic number 白名单

现在虽然 probe 了 detected_ext，但最好再加一层：

扩展名允许
+ MIME 类型允许
+ 文件头 magic 允许

原因是有人可能上传：

xxx.pdf 实际是 zip
xxx.docx 实际是空壳
xxx.jpg 实际是脚本

推荐规则：

如果 ext / mime / magic 三者明显冲突：
- 标记为 suspicious_file_type
- 默认拒绝入库
- 或进入 manual_review / parse_failed

尤其你现在支持 .json .log .csv .txt，这些纯文本类文件更容易被乱塞内容，入口要更稳一点。

3. 加密 / 损坏 / 空解析文件规则

需要显式识别：

PDF 加密
PDF 损坏
docx 损坏
xlsx 损坏
图片无法读取
解析后正文为空
解析后有效文本过少

不要让这些文件进入正常 indexing。

建议状态：

parse_failed
parse_empty
parse_low_quality
unsupported_or_corrupt
encrypted_file

现在如果只是解析失败变成任务失败，前端和用户很难知道原因。

二、文档身份规则要重做

这是当前最重要的修改点之一。

你现在的 source 很可能主要来自 filename。法规类 RAG 不能只靠 filename，因为文件名里往往包含：

标题
发布日期
施行日期
版本日期
括号说明
空格
下划线
上传批次

建议把“文件名”拆成四层身份。

1. original_filename

保留原始文件名，不能丢。

{
  "original_filename": "聊城市养犬管理条例_2020-06-15_2020-09-01.docx"
}

用于展示、下载、审计。

2. source_id

内部唯一 ID，不建议直接用文件名。

可以用：

hash(original_filename + content_sha256)

或者数据库自增 / uuid。

{
  "source_id": "doc_20260511_xxxxx"
}
3. canonical_title

从文件名或正文标题中抽出的规范标题。

{
  "canonical_title": "聊城市养犬管理条例"
}

这个是后续文档名不全检索的核心字段。

4. doc_version / effective_date / publish_date

法规文档必须显式抽版本信息。

例如：

{
  "publish_date": "2020-06-15",
  "effective_date": "2020-09-01",
  "doc_version_label": "2020-09-01施行版"
}

否则后面会出现：

同名法规多个版本
旧版残留
新版覆盖不清
用户问“现行版本”时无法判断
三、重复上传规则要补

现在你是每个 source 拿锁，但还需要区分几种重复情况。

1. 同文件名 + 同内容

应该直接返回：

already_exists / no_change

不要重复解析、重复 embedding。

判断依据：

original_filename 相同
content_sha256 相同
2. 同文件名 + 不同内容

这是重传 / 修订。

应该走：

reindexing
pending_version = next_version
旧 active_version 继续服务
新版本发布成功后切 active_version

你现在已经基本是这个逻辑。

但建议前端提示：

检测到同名文档内容变化，正在生成新版本。旧版本在新版本发布前继续可用。
3. 不同文件名 + 同 canonical_title

这是法规类高频情况。

例如：

聊城市养犬管理条例.docx
聊城市养犬管理条例_2020-09-01.docx
聊城养犬条例新版.pdf

它们可能其实是同一法规不同命名。

建议规则：

canonical_title 相同
但 original_filename 不同
→ 标记为 same_title_candidate
→ 如果版本日期相同且 hash 接近，可提示疑似重复
→ 如果版本日期不同，则作为同法规不同版本处理

这个非常关键。否则 source 会越来越乱，后面查询“养犬条例”时容易多文档竞争。

四、解析路由规则要从“文件级”升级到“页级 / 质量级”

你现在 PDF 分两路：

数字 PDF → pdf_digital_fast
扫描 / 乱码 / 图片页多 → pdf_ocr_layout

这个方向对，但建议不要只做文件级判断。

当前风险

很多 PDF 是混合型：

前几页可抽文本
后面是扫描页
正文可抽，表格是图片
目录可抽，正文乱码

如果只判断整个 PDF 走 digital 或 OCR，容易出现：

部分页面丢失
正文质量低
章节结构错乱
推荐规则：PDF 页级质量检测

给每页打标签：

{
  "page_no": 1,
  "text_extractable": true,
  "text_length": 1200,
  "garbled_ratio": 0.02,
  "image_area_ratio": 0.1,
  "route": "digital"
}

如果某页：

文本过少
乱码比例高
图片占比高

单页走 OCR。

最后合并为统一 Document IR。

也就是：

不是 pdf_digital_fast vs pdf_ocr_layout 二选一
而是 digital 优先，低质量页 OCR 补偿

这个对法规 PDF 很重要。

五、入库时必须生成“文档画像”

这是你现在最该加的规则。

你前面问“文档名字不全时怎么检索”，答案其实不应该主要发生在查询时，而应该在上传入库时就把文档画像建好。

每个文档上传后，需要生成：

{
  "canonical_title": "聊城市养犬管理条例",
  "title_aliases": [
    "聊城养犬条例",
    "聊城市养犬条例",
    "养犬管理条例",
    "养犬条例"
  ],
  "region": "聊城市",
  "doc_type": "条例",
  "publish_date": "2020-06-15",
  "effective_date": "2020-09-01",
  "section_titles": [
    "总则",
    "养犬登记",
    "养犬行为规范",
    "法律责任",
    "附则"
  ],
  "article_titles": [],
  "topic_terms": [
    "养犬行为规范",
    "犬只免疫",
    "禁养犬",
    "犬只登记",
    "法律责任",
    "处罚"
  ]
}

这部分应该写入独立表，例如：

document_profiles
document_aliases
document_sections
document_topics

不要只存在 chunk 里。

六、自动别名生成规则要前置到入库阶段

不要靠人工维护 alias 表。

你应该在上传成功解析标题后，自动生成别名。

规则一：行政区前缀弱化
聊城市养犬管理条例
→ 聊城养犬管理条例
→ 养犬管理条例
→ 养犬条例
陵水黎族自治县非物质文化遗产保护条例
→ 陵水非物质文化遗产保护条例
→ 陵水非遗保护条例
→ 非物质文化遗产保护条例
→ 非遗保护条例
规则二：法规文种保留

保留这些尾词：

条例
办法
规定
细则
决定
通知
意见
规则
规程
标准

用户问：

养犬条例
非遗条例
河道管理规定
地方立法条例

都要能匹配。

规则三：长词简称替换

建立少量领域通用简称表，不是每个文档人工配。

例如：

{
  "非物质文化遗产": ["非遗"],
  "城市管理行政执法": ["城管执法"],
  "人民代表大会": ["人大"],
  "地方性法规": ["地方立法", "法规"]
}

这类表是通用的，不是按文档新增，所以维护成本可控。

规则四：章节标题进入别名候选

如果文档里有章节：

第三章 养犬行为规范

那么用户问：

养犬行为规范有哪些要求？

应该可以反推：

聊城市养犬管理条例

所以 section_title 不是普通 chunk 文本，它要进入 source resolution。

七、分块规则需要更贴合法规结构

你现在已经有 Document IR 和分块，但法规类文档建议强化以下规则。

1. 标题 / 目录 / 正文章节分开处理

不要让这些 chunk 权重一样：

文档标题
目录
章节标题
正文条款
附则
附件

尤其目录 chunk 和标题 chunk 容易在检索里上浮，造成“看起来命中文档，但没有证据”。

建议 chunk_type：

title
toc
chapter_heading
section_heading
article
paragraph
table
appendix

检索时：

title / heading 可用于 source resolution
article / paragraph 用于 answer evidence
toc 默认降权
2. 章节标题不能单独作为最终证据

例如命中：

第三章 养犬行为规范

不能直接回答“有规定”。

它只能作为路由线索，然后继续找该章节下属条款。

规则：

heading_hit → expand_to_child_articles
3. 条文要保留 article_no

法规回答经常需要引用：

第十七条
第二十一条

chunk metadata 应有：

{
  "chapter": "第三章 养犬行为规范",
  "article_no": "第十七条",
  "section_path": "第三章 养犬行为规范 > 第十七条"
}

否则回答引用会虚。

4. 父子关系要入库

建议保存：

chapter → section → article → paragraph

至少要有：

{
  "parent_heading": "第三章 养犬行为规范",
  "prev_chunk_id": "...",
  "next_chunk_id": "...",
  "child_range": ["chunk_12", "chunk_13", "chunk_14"]
}

这样用户问章节级问题时，可以从章节扩展到正文条款。

八、SQLite FTS 和 Milvus 的写入内容要区分

你现在说：

分块、补上下文、批量 embedding、写 Milvus，同时把 chunk 文本写进 SQLite FTS。

这里要注意一个规则：

Milvus 可以写 contextual_text
SQLite FTS 最好写 raw_text + title/section 字段

不要把“补上下文后的长文本”直接塞进 FTS 主召回，否则关键词命中会变脏。

推荐：

{
  "raw_text": "第十七条 养犬人应当...",
  "contextual_text": "《聊城市养犬管理条例》第三章 养犬行为规范。第十七条...",
  "fts_text": "第十七条 养犬人应当...",
  "embedding_text": "《聊城市养犬管理条例》第三章 养犬行为规范。第十七条..."
}

也就是：

FTS 用原文精准召回
向量用上下文增强召回

你之前 RAG 项目里已经踩过这个点，这里要继续保持。

九、发布前要加质量闸门

你现在是：

SQLite 事务提交
→ vector_pending
→ worker 确认 Milvus 可见
→ active_version
→ completed

这个整体对，但“确认 Milvus 可见”不应该只看有无记录。

建议发布前检查：

1. SQLite chunk_count > 0
2. Milvus vector_count == expected_chunk_count
3. source + pending_version 在 Milvus 中可查询
4. document_profile 已生成
5. section_index 已生成
6. parse_quality_score 达标
7. embedding batch 没有缺失
8. FTS 行数与 chunks_meta 行数一致

可以形成 publish gate：

{
  "publish_gate": {
    "sqlite_chunks_ok": true,
    "fts_chunks_ok": true,
    "milvus_vectors_ok": true,
    "profile_ok": true,
    "section_index_ok": true,
    "parse_quality_ok": true
  }
}

只有全部通过，才能：

pending_version → active_version
status → completed

否则：

status → vector_failed / index_failed / profile_failed / parse_low_quality
旧 active_version 继续服务
十、状态语义要细化

你现在的状态有：

accepted
reindexing
vector_pending
vector_failed
completed
deleting
pending_delete
delete_failed
not_found

建议补充：

validating
parsing
parse_failed
parse_empty
parse_low_quality
chunking
embedding
indexing_sqlite
indexing_vector
profile_building
publish_pending
publish_failed
completed

不一定都对外展示，但内部最好有。

前端可压缩展示成：

排队中
解析中
索引中
发布中
已完成
失败

但是内部日志一定要细。

十一、上传接口返回语义要改

你提到：

上传接口返回的 completed 只表示这次任务处理完成到 vector_pending/提交阶段；真正对问答检索稳定可见，还要等补偿 worker 完成版本发布。

这个语义容易误导。

建议不要把上传接口的 task completed 和文档 searchable completed 混用。

拆成两个字段：

{
  "task_status": "completed",
  "document_status": "vector_pending",
  "searchable": false,
  "active_version": 3,
  "pending_version": 4
}

等 worker 发布成功后：

{
  "task_status": "completed",
  "document_status": "completed",
  "searchable": true,
  "active_version": 4,
  "pending_version": null
}

前端显示：

上传处理完成，正在发布索引，旧版本仍可检索

而不是直接显示“完成”。

十二、删除 / 重传期间的可见性规则要更明确

你现在：

deleting / pending_delete / delete_failed 整体隐藏 source

这块要小心。

建议分两种删除：

1. 用户主动删除文档

应该立即隐藏：

visibility = hidden
status = deleting

即使 Milvus 删除还没完成，也不允许检索出来。

这个你现在方向对。

2. 重传 / 重建文档

不能隐藏旧版本。

应该：

旧 active_version 继续服务
新 pending_version 不可见

这个你现在也对。

建议状态明确区分：

reindexing，不隐藏旧版本
deleting，隐藏全部版本
十三、需要新增 doc_index / section_index

这是解决你现在“文档名不全、章节没命中”的关键。

你现在有：

documents
documents_fts
chunks_meta
chunks_fts
Milvus

建议再加：

document_profiles
document_aliases
document_sections
document_topics

用途分别是：

document_profiles：文档级画像
document_aliases：标题别名、简称、弱引用
document_sections：章节标题、层级、对应 chunk 范围
document_topics：从标题和正文抽出的主题词

查询时先跑：

source_resolver

它查这些表，而不是一上来查 chunk。

十四、文档名不全时的入库规则支持

为了后续查询能处理：

非遗条例奖励处罚
养犬行为规范有哪些要求
地方立法条例立法程序
河道管理处罚规定

入库阶段必须生成：

{
  "source_resolution_fields": {
    "canonical_title": "...",
    "title_aliases": [],
    "region_terms": [],
    "doc_type_terms": [],
    "section_titles": [],
    "topic_terms": [],
    "date_terms": []
  }
}

然后查询时才能做：

标题残缺匹配
简称匹配
章节反推文档
主题反推文档
版本日期匹配

否则你只能在查询时临时猜，稳定性不会高。

十五、建议修改后的完整上传入库流程

可以改成这样：

1. upload accepted
2. 基础校验
   - filename
   - content
   - size
   - extension
   - mime
   - magic
   - hash
3. 重复检测
   - same file same hash
   - same title same version
   - same title different version
4. 创建 task
5. 写控制面状态 validating / accepted
6. 文件 probe
7. parser 路由
8. Document IR 解析
9. parse quality 评估
10. 生成 document_profile
    - canonical_title
    - aliases
    - region
    - doc_type
    - dates
    - section_titles
    - topic_terms
11. 法规结构化切块
    - title
    - toc
    - chapter
    - article
    - paragraph
12. pending_version 重建
13. SQLite 事务写入
    - chunks_meta
    - chunks_fts
    - document_profiles
    - document_aliases
    - document_sections
14. Milvus 写入 pending_version
15. 状态进入 publish_pending / vector_pending
16. publish worker 检查质量闸门
17. 通过后：
    - pending_version → active_version
    - completed
    - searchable = true
18. 失败则：
    - publish_failed / vector_failed
    - 旧 active_version 继续服务
19. 异步清理旧版本