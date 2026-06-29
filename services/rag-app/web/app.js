// RAG 系统前端 JavaScript

// 配置
function normalizeBaseUrl(url) {
    return (url || '').trim().replace(/\/+$/, '');
}

const DEFAULT_API_BASE_URL = (typeof window !== 'undefined'
    && window.location
    && window.location.origin
    && window.location.origin !== 'null')
    ? window.location.origin
    : 'http://localhost:8080';

let API_BASE_URL = normalizeBaseUrl(localStorage.getItem('apiBaseUrl')) || DEFAULT_API_BASE_URL;
const CHAT_USER_ID_KEY = 'ragChatUserId';

function getOrCreateChatUserId() {
    let userId = (localStorage.getItem(CHAT_USER_ID_KEY) || '').trim();
    if (!userId) {
        userId = `web_${Date.now()}_${Math.floor(Math.random() * 100000)}`;
        localStorage.setItem(CHAT_USER_ID_KEY, userId);
    }
    return userId;
}

function resolveApiBaseUrl() {
    const origin = DEFAULT_API_BASE_URL;
    const base = normalizeBaseUrl(API_BASE_URL);
    if (base && base.includes("localhost") && origin && !origin.includes("localhost")) {
        return "";
    }
    if (!base) {
        return "";
    }
    if (origin && base === normalizeBaseUrl(origin)) {
        return "";
    }
    return base;
}

function getEffectiveApiBaseUrl() {
    return resolveApiBaseUrl() || normalizeBaseUrl(DEFAULT_API_BASE_URL);
}

function updateApiBaseStatus() {
    if (elements.currentPageOrigin) {
        elements.currentPageOrigin.textContent = normalizeBaseUrl(DEFAULT_API_BASE_URL) || '-';
    }
    if (elements.effectiveApiBase) {
        elements.effectiveApiBase.textContent = getEffectiveApiBaseUrl() || '-';
    }
}

// 状态
let state = {
    currentTab: 'chat',
    messages: [],
    documents: [],
    stats: {
        totalQueries: 0,
        avgResponseTime: 0,
        totalDocs: 0,
        satisfaction: 0
    }
};

// DOM 元素
const elements = {
    navItems: document.querySelectorAll('.nav-item'),
    tabContents: document.querySelectorAll('.tab-content'),
    questionInput: document.getElementById('question-input'),
    sendButton: document.getElementById('send-button'),
    messagesContainer: document.getElementById('messages-container'),
    loadingOverlay: document.getElementById('loading-overlay'),
    uploadButton: document.getElementById('upload-button'),
    fileUploadInput: document.getElementById('file-upload-input'),
    documentsList: document.getElementById('documents-list'),
    apiUrlInput: document.getElementById('api-base-url'),
    currentPageOrigin: document.getElementById('current-page-origin'),
    effectiveApiBase: document.getElementById('effective-api-base'),
    resetApiBaseButton: document.getElementById('reset-api-base'),
    topKInput: document.getElementById('top-k'),
    rerankSelect: document.getElementById('rerank-enabled')
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadSettings();
    loadDocuments();
    updateStats();
});

// 事件监听器
function initializeEventListeners() {
    // Tab 切换
    elements.navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = item.dataset.tab;
            switchTab(tabId);
        });
    });

    // 发送消息
    elements.sendButton.addEventListener('click', sendMessage);
    
    elements.questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 自动调整输入框高度
    elements.questionInput.addEventListener('input', autoResize);

    // 上传按钮
    elements.uploadButton.addEventListener('click', showUploadDialog);
    elements.fileUploadInput.addEventListener('change', handleFileSelected);

    // API 设置保存
    elements.apiUrlInput.addEventListener('change', saveSettings);
    elements.topKInput.addEventListener('change', saveSettings);
    elements.rerankSelect.addEventListener('change', saveSettings);
    elements.resetApiBaseButton.addEventListener('click', resetApiBaseUrl);
}

function resetApiBaseUrl() {
    const origin = normalizeBaseUrl(DEFAULT_API_BASE_URL);
    if (elements.apiUrlInput) {
        elements.apiUrlInput.value = origin;
    }
    saveSettings();
}

// Tab 切换
function switchTab(tabId) {
    state.currentTab = tabId;
    
    // 更新导航项
    elements.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.tab === tabId);
    });

    // 更新内容区
    elements.tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabId}-tab`);
    });
}

// 自动调整输入框高度
function autoResize() {
    const textarea = elements.questionInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

// 发送消息
async function sendMessage() {
    const question = elements.questionInput.value.trim();
    if (!question) return;

    // 添加用户消息
    addMessage(question, 'user');
    elements.questionInput.value = '';
    autoResize();

    // 显示加载状态
    setLoading(true);

    try {
        // 调用 API
        const startTime = Date.now();
        const response = await fetch(`${resolveApiBaseUrl()}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: question,
                user_id: getOrCreateChatUserId(),
                top_k: parseInt(elements.topKInput.value) || 10,
                enable_rerank: elements.rerankSelect.value === 'true'
            })
        });

        const data = await response.json();
        const responseTime = ((Date.now() - startTime) / 1000).toFixed(2);

        const meta = (data && data.metadata) || {};
        const refused = meta.refused;
        if (refused) {
            const firstSrc = (data.sources && data.sources[0] && data.sources[0].source) || '';
            const candidates = (meta.candidate_sources || []).filter(Boolean);
            let msg = '';
            let showAsAssistant = false;
            if (
                (meta.final_channel === 'document_clarification' ||
                    meta.answer_mode === 'clarification' ||
                    refused === 'tier2_soft_confirm' ||
                    refused === 'tier3_summary_clarification' ||
                    refused === 'document_clarification') &&
                (data.answer || meta.clarification)
            ) {
                msg = data.answer || meta.clarification;
                showAsAssistant = true;
            } else if (refused === 'retrieval_error') {
                msg = '知识库检索暂时不可用，请稍后重试。';
            } else if (refused === 'section_anchor_ambiguous') {
                if (candidates.length > 0) {
                    msg = `请确认要查询哪一部法规：\n${candidates.map((name, idx) => `${idx + 1}. ${name}`).join('\n')}`;
                } else {
                    msg = '检测到多个可能命中的法规，请先指定文档名称后再查询。';
                }
            } else if (refused === 'document_target_required') {
                msg = (data && data.answer) || '请先说明要查询哪一部法规文档，再继续检索。';
            } else if (refused === 'doc_found_but_no_structured_stats') {
                msg = firstSrc
                    ? `已定位到文档 ${firstSrc}，但当前知识库未保存该文件的结构化统计信息，暂无法直接回答数量类问题。`
                    : '已定位到文档，但当前知识库未保存该文件的结构化统计信息。';
            } else if (refused === 'low_relevance_filtered') {
                msg = '检索到的证据相关性过低，已被过滤。';
            } else if (refused === 'no_relevant_evidence') {
                msg = '未检索到相关证据。';
            } else {
                msg = '未在知识库中找到足够相关的证据来回答该问题。';
            }
            if (showAsAssistant) {
                addAssistantMessage(msg, data.sources || [], responseTime, data.metadata || {});
            } else {
                addErrorMessage(msg);
            }
        } else {
            addAssistantMessage(data.answer, data.sources, responseTime, data.metadata || {});
        }

        // 更新统计
        updateStats();

    } catch (error) {
        console.error('Error:', error);
        addErrorMessage('发送失败，请检查网络连接或 API 配置');
    } finally {
        setLoading(false);
    }
}

// 添加消息
function addMessage(content, role) {
    const message = {
        id: Date.now(),
        content: content,
        role: role,
        timestamp: new Date().toISOString()
    };

    state.messages.push(message);
    renderMessage(message);
    scrollToBottom();
}

// 渲染消息
function renderMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;
    messageDiv.id = `message-${message.id}`;

    const avatarIcon = message.role === 'user' ? 'fa-user' : 'fa-robot';
    const senderName = message.role === 'user' ? '我' : 'AI 助手';
    const timestamp = formatTimestamp(message.timestamp);

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas ${avatarIcon}"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender-name">${senderName}</span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-text">
                <p>${escapeHtml(message.content)}</p>
            </div>
        </div>
    `;

    elements.messagesContainer.appendChild(messageDiv);
    enhanceCitationRefs(messageDiv, message);
}

// 添加助手消息（带来源）
function addAssistantMessage(answer, sources, responseTime, metadata = {}) {
    const message = {
        id: Date.now(),
        content: answer,
        role: 'assistant',
        sources: sources,
        timestamp: new Date().toISOString(),
        responseTime: responseTime,
        serverTiming: (metadata && metadata.server_timing_ms) || null
    };

    state.messages.push(message);
    renderAssistantMessage(message);
    scrollToBottom();
}

// 渲染助手消息
function renderAssistantMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;
    messageDiv.id = `message-${message.id}`;

    const timestamp = formatTimestamp(message.timestamp);
    const serverTiming = formatServerTiming(message.serverTiming);
    const answerHtml = renderAssistantMarkdown(message.content);

    const sourcesHtml = buildEvidenceSection(message);

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender-name">AI 助手</span>
                <span class="timestamp">${timestamp} (响应：${message.responseTime}s)</span>
            </div>
            <div class="message-text assistant-answer-text">
                ${answerHtml}
            </div>
            ${sourcesHtml}
        </div>
    `;

    elements.messagesContainer.appendChild(messageDiv);
    enhanceCitationRefs(messageDiv, message);
}

function formatServerTiming(serverTiming) {
    if (!serverTiming || typeof serverTiming !== 'object') {
        return '';
    }

    const parts = [];
    if (typeof serverTiming.total_request === 'number') {
        parts.push(`总计 ${serverTiming.total_request.toFixed(0)}ms`);
    }
    if (typeof serverTiming.recall === 'number') {
        parts.push(`检索 ${serverTiming.recall.toFixed(0)}ms`);
    }
    if (typeof serverTiming.pre_answer === 'number') {
        parts.push(`整理 ${serverTiming.pre_answer.toFixed(0)}ms`);
    }
    if (typeof serverTiming.answer === 'number') {
        parts.push(`生成 ${serverTiming.answer.toFixed(0)}ms`);
    }
    if (typeof serverTiming.handler_total === 'number' && typeof serverTiming.total_request !== 'number') {
        parts.push(`总计 ${serverTiming.handler_total.toFixed(0)}ms`);
    }

    return parts.join(' / ');
}

// 添加错误消息
function addErrorMessage(error) {
    addMessage(error, 'error');
}

// 滚动到底部
function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// 加载状态
function setLoading(loading) {
    elements.loadingOverlay.classList.toggle('active', loading);
    elements.sendButton.disabled = loading;
}

// 格式化时间戳
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
    
    return date.toLocaleString('zh-CN');
}

// HTML 转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getEvidenceSources(message) {
    return Array.isArray(message.sources) ? message.sources : [];
}

function getSourceRef(source, index) {
    const ref = source && source.ref !== undefined && source.ref !== null ? source.ref : index + 1;
    return String(ref).replace(/^\[|\]$/g, '');
}

function getSafeEvidenceId(messageId, ref) {
    return `evidence-${String(messageId).replace(/[^\w-]/g, '')}-${String(ref).replace(/[^\w-]/g, '')}`;
}

function getSourceMeta(source) {
    return (source && (source.metadata || source.meta)) || {};
}

function getSourceText(source) {
    return String((source && (source.text || source.content || source.chunk_text)) || '').trim();
}

function compactText(text, maxLength = 110) {
    const normalized = String(text || '').replace(/\s+/g, ' ').trim();
    if (normalized.length <= maxLength) {
        return normalized;
    }
    return `${normalized.slice(0, maxLength).trim()}...`;
}

function getEvidenceChunkLabel(source) {
    const meta = getSourceMeta(source);
    const chunkValue = source && (source.chunk_index ?? source.chunk_id ?? meta.chunk_index ?? meta.chunk_id);
    const totalValue = source && (source.chunk_count ?? source.total_chunks ?? meta.chunk_count ?? meta.total_chunks);
    const chunkNumber = Number(chunkValue);
    const totalNumber = Number(totalValue);

    if (Number.isFinite(chunkNumber) && Number.isFinite(totalNumber) && totalNumber > 0) {
        const displayChunk = chunkNumber >= 0 && chunkNumber < totalNumber ? chunkNumber + 1 : chunkNumber;
        return `第 ${displayChunk}/${totalNumber} 块`;
    }
    if (Number.isFinite(chunkNumber)) {
        return `第 ${chunkNumber + 1} 块`;
    }
    const chunkRange = String((source && source.chunk_range) || meta.chunk_range || '').trim();
    if (chunkRange) {
        const parts = chunkRange.split('-').map(part => Number(part.trim()));
        if (parts.length === 2 && parts.every(Number.isFinite)) {
            return `第 ${parts[0] + 1}-${parts[1] + 1} 块`;
        }
        if (parts.length === 1 && Number.isFinite(parts[0])) {
            return `第 ${parts[0] + 1} 块`;
        }
        return `第 ${chunkRange} 块`;
    }
    return '证据片段';
}

function getEvidenceSummary(source) {
    const meta = getSourceMeta(source);
    const explicit = meta.section_title || meta.section || meta.title || source?.section_title || source?.section;
    if (explicit) {
        return compactText(explicit, 36);
    }
    return compactText(getSourceText(source), 42) || '原文片段';
}

function buildEvidenceSection(message) {
    const sources = getEvidenceSources(message);
    if (sources.length === 0) {
        return '';
    }

    const documentNames = new Set(sources.map(source => source.source || source.document || '未知文档'));
    const chips = sources
        .map((source, index) => `<span class="evidence-chip">[${escapeHtml(getSourceRef(source, index))}]</span>`)
        .join('');
    const grouped = sources.reduce((groups, source, index) => {
        const documentName = source.source || source.document || '未知文档';
        if (!groups.has(documentName)) {
            groups.set(documentName, []);
        }
        groups.get(documentName).push({ source, index });
        return groups;
    }, new Map());

    const documentBlocks = Array.from(grouped.entries()).map(([documentName, items]) => {
        const cards = items.map(({ source, index }) => {
            const ref = getSourceRef(source, index);
            const text = getSourceText(source);
            return `
                <details class="evidence-card" id="${getSafeEvidenceId(message.id, ref)}" data-evidence-ref="${escapeHtml(ref)}">
                    <summary class="evidence-card-summary">
                        <span class="evidence-ref">[${escapeHtml(ref)}]</span>
                        <span>${escapeHtml(getEvidenceChunkLabel(source))}</span>
                        <span class="evidence-summary-text">${escapeHtml(getEvidenceSummary(source))}</span>
                        <span class="evidence-card-action" aria-hidden="true"></span>
                    </summary>
                    <div class="evidence-full">
                        <div class="evidence-full-label">原文片段</div>
                        <p>${escapeHtml(text)}</p>
                    </div>
                </details>
            `;
        }).join('');

        return `
            <section class="evidence-document">
                <h4>${escapeHtml(documentName)}</h4>
                ${cards}
            </section>
        `;
    }).join('');

    return `
        <details class="evidence-section">
            <summary class="evidence-summary">
                <span class="evidence-summary-title">证据来源 ${sources.length} 条｜来自 ${documentNames.size} 份文档</span>
                <span class="evidence-summary-chips">${chips}</span>
            </summary>
            <div class="evidence-body">
                ${documentBlocks}
            </div>
        </details>
    `;
}

function enhanceCitationRefs(messageDiv, message) {
    const sources = getEvidenceSources(message);
    if (sources.length === 0) {
        return;
    }

    const sourcesByRef = new Map(sources.map((source, index) => [getSourceRef(source, index), source]));
    const answer = messageDiv.querySelector('.assistant-answer-text');
    if (!answer) {
        return;
    }

    const walker = document.createTreeWalker(answer, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            const parent = node.parentElement;
            if (!parent || parent.closest('a, button, code, pre, .citation-ref')) {
                return NodeFilter.FILTER_REJECT;
            }
            return /\[(\d+)\]/.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
    });

    const textNodes = [];
    while (walker.nextNode()) {
        textNodes.push(walker.currentNode);
    }

    textNodes.forEach(node => {
        const fragment = document.createDocumentFragment();
        const value = node.nodeValue;
        let lastIndex = 0;
        value.replace(/\[(\d+)\]/g, (match, ref, offset) => {
            if (!sourcesByRef.has(ref)) {
                return match;
            }
            fragment.appendChild(document.createTextNode(value.slice(lastIndex, offset)));
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'citation-ref';
            button.dataset.ref = ref;
            button.textContent = match;
            button.addEventListener('mouseenter', () => showCitationPreview(button, sourcesByRef.get(ref)));
            button.addEventListener('focus', () => showCitationPreview(button, sourcesByRef.get(ref)));
            button.addEventListener('mouseleave', hideCitationPreview);
            button.addEventListener('blur', hideCitationPreview);
            button.addEventListener('click', () => focusEvidence(messageDiv, message.id, ref));
            fragment.appendChild(button);
            lastIndex = offset + match.length;
            return match;
        });
        fragment.appendChild(document.createTextNode(value.slice(lastIndex)));
        node.parentNode.replaceChild(fragment, node);
    });
}

function getCitationPreviewElement() {
    let preview = document.querySelector('.citation-preview');
    if (!preview) {
        preview = document.createElement('div');
        preview.className = 'citation-preview';
        preview.setAttribute('role', 'tooltip');
        document.body.appendChild(preview);
    }
    return preview;
}

function showCitationPreview(button, source) {
    if (!source) {
        return;
    }
    const preview = getCitationPreviewElement();
    const ref = button.dataset.ref;
    const documentName = source.source || source.document || '未知文档';
    preview.innerHTML = `
        <div class="citation-preview-title">[${escapeHtml(ref)}] ${escapeHtml(getEvidenceChunkLabel(source))}</div>
        <div class="citation-preview-source">${escapeHtml(documentName)}</div>
        <p>${escapeHtml(compactText(getSourceText(source), 180))}</p>
    `;

    const rect = button.getBoundingClientRect();
    const preferredTop = rect.top - preview.offsetHeight - 12;
    const top = preferredTop >= 12 ? preferredTop : rect.bottom + 10;
    const left = Math.min(window.innerWidth - preview.offsetWidth - 12, Math.max(12, rect.left));
    preview.style.top = `${top}px`;
    preview.style.left = `${left}px`;
    preview.classList.add('visible');
}

function hideCitationPreview() {
    const preview = document.querySelector('.citation-preview');
    if (preview) {
        preview.classList.remove('visible');
    }
}

function focusEvidence(messageDiv, messageId, ref) {
    hideCitationPreview();
    const section = messageDiv.querySelector('.evidence-section');
    if (section) {
        section.open = true;
    }

    const card = messageDiv.querySelector(`#${getSafeEvidenceId(messageId, ref)}`);
    if (!card) {
        return;
    }
    card.open = true;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('is-targeted');
    window.setTimeout(() => card.classList.remove('is-targeted'), 1600);
}

function formatAssistantInline(text) {
    let safe = escapeHtml(text || '');
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
    return safe;
}

function renderAssistantMarkdown(content) {
    const normalized = String(content || '').replace(/\r\n?/g, '\n').trim();
    if (!normalized) {
        return '<p></p>';
    }

    if (
        window.marked &&
        typeof window.marked.parse === 'function' &&
        window.DOMPurify &&
        typeof window.DOMPurify.sanitize === 'function'
    ) {
        window.marked.setOptions({
            gfm: true,
            breaks: true
        });

        const rawHtml = window.marked.parse(normalized);
        return window.DOMPurify.sanitize(rawHtml, {
            USE_PROFILES: { html: true }
        });
    }

    return renderAssistantStructuredText(normalized);
}

function normalizeStructuredCitations(value) {
    const candidates = Array.isArray(value) ? value : [value];
    const refs = [];
    for (const candidate of candidates) {
        if (candidate === null || candidate === undefined) {
            continue;
        }
        const matches = String(candidate).match(/\d+/g) || [];
        for (const raw of matches) {
            const ref = Number(raw);
            if (Number.isInteger(ref) && ref > 0 && !refs.includes(ref)) {
                refs.push(ref);
            }
        }
    }
    return refs;
}

function renderAssistantStructuredText(content) {
    const normalized = String(content || '').replace(/\r\n?/g, '\n').trim();
    if (!normalized) {
        return '<p></p>';
    }

    const lines = normalized.split('\n');
    const html = [];
    let listItems = [];

    function flushList() {
        if (!listItems.length) {
            return;
        }
        html.push(`<ul>${listItems.join('')}</ul>`);
        listItems = [];
    }

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            flushList();
            continue;
        }

        const bulletMatch = line.match(/^([-*]|\d+\.)\s+(.+)$/);
        if (bulletMatch) {
            listItems.push(`<li>${formatAssistantInline(bulletMatch[2])}</li>`);
            continue;
        }

        flushList();

        const headingMatch = line.match(/^(#{2,6})\s+(.+)$/);
        if (headingMatch) {
            const level = Math.min(3, Math.max(2, headingMatch[1].length));
            const tagName = level === 2 ? 'h2' : 'h3';
            html.push(`<${tagName} class="message-heading message-heading-${level}">${formatAssistantInline(headingMatch[2])}</${tagName}>`);
            continue;
        }

        if (line.startsWith('方面：') || line.startsWith('未覆盖：')) {
            html.push(`<h3 class="message-heading message-heading-legacy">${formatAssistantInline(line)}</h3>`);
            continue;
        }

        html.push(`<p>${formatAssistantInline(line)}</p>`);
    }

    flushList();
    return html.join('');
}

// 加载文档列表
async function loadDocuments() {
    try {
        const response = await fetch(`${resolveApiBaseUrl()}/documents`);
        const data = await response.json();
        
        state.documents = data.documents || [];
        renderDocuments();
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

// 渲染文档列表
function renderDocuments() {
    if (state.documents.length === 0) {
        elements.documentsList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-folder-open"></i>
                <p>暂无文档</p>
                <p class="empty-hint">点击"上传文档"开始添加知识库</p>
            </div>
        `;
        return;
    }

    elements.documentsList.innerHTML = state.documents.map(doc => {
        const created = doc.created_at ? new Date(doc.created_at).toLocaleString('zh-CN') : '-';
        const status = doc.status || '-';
        const isCompleted = ['completed', 'indexed'].includes(status);
        const chunks = (isCompleted && doc.chunks_indexed != null) ? doc.chunks_indexed : '';
        const error = (status === 'failed' && doc.error) ? doc.error : '';
        const canView = isCompleted;
        const viewBtn = canView
            ? `<button class="btn btn-secondary" onclick="viewDocument('${encodeURIComponent(doc.filename)}')">
                   <i class="fas fa-eye"></i> 查看
               </button>`
            : `<button class="btn btn-secondary" disabled title="索引中或失败，暂不可查看">
                   <i class="fas fa-eye"></i> 查看
               </button>`;
        const retryBtn = (status === 'failed' && doc.task_id)
            ? `<button class="btn btn-primary" onclick="retryTask('${doc.task_id}')">
                   <i class="fas fa-rotate-right"></i> 重试
               </button>`
            : '';
        const statusBadge = `<span class="badge badge-${status}">${status}</span>`;
        const extraInfo = isCompleted
            ? `<span class="doc-extra">分块：${chunks}</span>`
            : status === 'failed'
                ? `<span class="doc-extra" style="color:#c00;">错误：${escapeHtml(error)}</span>`
                : `<span class="doc-extra">索引进行中</span>`;
        return `
        <div class="document-item">
            <div class="document-icon">
                <i class="fas fa-file-alt"></i>
            </div>
            <div class="document-info">
                <div class="document-name">${escapeHtml(doc.filename)}</div>
                <div class="document-meta">
                    上传时间：${created} · 状态：${statusBadge}
                </div>
                <div class="document-meta">${extraInfo}</div>
            </div>
            <div class="document-actions">
                ${viewBtn}
                ${retryBtn}
                <button class="btn btn-danger" onclick="deleteDocument('${doc.filename}')">
                    <i class="fas fa-trash"></i> 删除
                </button>
            </div>
        </div>`;
    }).join('');
}

// 显示上传对话框
function showUploadDialog() {
    elements.fileUploadInput.click();
}

async function handleFileSelected(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    const results = await uploadFiles(files, 2);
    // 每个上传返回 task_id，启动轮询更新列表
    for (const r of results) {
        if (r && r.task_id) {
            pollTask(r.task_id, async () => {
                await loadDocuments();
            }).catch(() => {});
        }
    }
    event.target.value = '';
}

// 上传文档
async function uploadDocument(file) {
    setLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${resolveApiBaseUrl()}/documents/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            return result;
        } else {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || '上传失败');
        }
    } catch (error) {
        console.error('Error uploading document:', error);
        alert(`上传失败：${error.message}`);
    } finally {
        setLoading(false);
    }
}

async function uploadWithRetry(file, retries = 3, delayMs = 800) {
    let attempt = 0;
    let d = delayMs;
    for (;;) {
        const res = await uploadDocument(file);
        if (res) return res;
        if (attempt >= retries) throw new Error('上传失败');
        await new Promise(r => setTimeout(r, d));
        d = Math.min(d * 2, 4000);
        attempt++;
    }
}

async function uploadFiles(files, concurrency = 2) {
    const queue = [...files];
    const running = new Set();
    const results = [];
    async function worker(f) {
        const r = await uploadWithRetry(f);
        results.push(r);
    }
    while (queue.length) {
        while (running.size < concurrency && queue.length) {
            const f = queue.shift();
            const p = worker(f).finally(() => running.delete(p));
            running.add(p);
        }
        await Promise.race([...running]);
    }
    await Promise.all([...running]);
    await loadDocuments();
    await updateStats();
    return results;
}

async function pollTask(taskId, onUpdate, intervalMs = 1500, maxMs = 5 * 60 * 1000) {
    const start = Date.now();
    for (;;) {
        const res = await fetch(`${resolveApiBaseUrl()}/tasks/${taskId}`);
        if (!res.ok) throw new Error('任务查询失败');
        const data = await res.json();
        try { onUpdate?.(data); } catch {}
        if (['completed', 'indexed', 'failed'].includes(data.status)) return data;
        if (Date.now() - start > maxMs) throw new Error('任务轮询超时');
        await new Promise(r => setTimeout(r, intervalMs));
    }
}

async function retryTask(taskId) {
    const res = await fetch(`${resolveApiBaseUrl()}/documents/${taskId}/retry`, { method: 'POST' });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`重试失败：${(err && err.detail) || '未知错误'}`);
        return;
    }
    const data = await res.json();
    pollTask(taskId, async () => {
        await loadDocuments();
    }).catch(() => {});
}

// 查看文档（真实详情）
async function viewDocument(encodedFilename) {
    const filename = decodeURIComponent(encodedFilename);

    try {
        const response = await fetch(`${resolveApiBaseUrl()}/documents/${encodeURIComponent(filename)}`);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || '获取文档详情失败');
        }

        const doc = await response.json();
        showDocumentModal(doc);
    } catch (error) {
        console.error('Error viewing document:', error);
        alert(`查看失败：${error.message}`);
    }
}

function showDocumentModal(doc) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.55);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        padding: 16px;
    `;

    const modal = document.createElement('div');
    modal.style.cssText = `
        width: min(980px, 100%);
        max-height: 90vh;
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 12px 36px rgba(0,0,0,0.25);
        overflow: hidden;
        display: flex;
        flex-direction: column;
    `;

    const content = escapeHtml(doc.content || '');

    modal.innerHTML = `
        <div style="padding: 14px 18px; border-bottom: 1px solid #eee; display:flex; justify-content:space-between; align-items:center; gap:10px;">
            <div>
                <div style="font-weight: 600; font-size: 1rem;">${escapeHtml(doc.filename || '文档详情')}</div>
            </div>
            <button id="doc-modal-close" style="border:0; background:transparent; font-size:1.25rem; cursor:pointer;">×</button>
        </div>
        <div style="padding: 18px; overflow:auto; background:#fff;">
            <div style="white-space:pre-wrap; color:#1f2937; font-size:0.95rem; line-height:1.75;">
                ${content || '<span style="color:#94a3b8;">（无内容）</span>'}
            </div>
        </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    const close = () => overlay.remove();
    modal.querySelector('#doc-modal-close').addEventListener('click', close);
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) close();
    });
}

// 删除文档（占位）
async function deleteDocument(filename) {
    if (!confirm(`确定要删除文档 "${filename}" 吗？`)) return;
    try {
        const res = await fetch(`${resolveApiBaseUrl()}/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || '删除失败');
        }
        alert('删除成功');
        await loadDocuments();
        await updateStats();
    } catch (e) {
        alert(`删除失败：${e.message}`);
    }
}

// 更新统计
async function updateStats() {
    try {
        // 这里应该从 API 获取真实数据
        const queryCount = state.messages.filter(m => m.role === 'assistant').length;
        const responseTimes = state.messages
            .filter(m => m.responseTime)
            .map(m => parseFloat(m.responseTime));
        const avgResponse = responseTimes.length > 0 
            ? (responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length).toFixed(2)
            : '0';

        state.stats = {
            totalQueries: queryCount,
            avgResponseTime: avgResponse,
            totalDocs: state.documents.length,
            satisfaction: 85 // 示例数据
        };

        // 更新 UI
        document.getElementById('total-queries').textContent = state.stats.totalQueries;
        document.getElementById('avg-response').textContent = `${state.stats.avgResponseTime}s`;
        document.getElementById('total-docs').textContent = state.stats.totalDocs;
        document.getElementById('satisfaction').textContent = `${state.stats.satisfaction}%`;

    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// 保存设置
function saveSettings() {
    const settings = {
        apiBaseUrl: normalizeBaseUrl(elements.apiUrlInput.value),
        topK: elements.topKInput.value,
        rerankEnabled: elements.rerankSelect.value
    };

    localStorage.setItem('ragSettings', JSON.stringify(settings));
    localStorage.setItem('apiBaseUrl', settings.apiBaseUrl);
    API_BASE_URL = normalizeBaseUrl(settings.apiBaseUrl) || API_BASE_URL;
    updateApiBaseStatus();
}

// 加载设置
function loadSettings() {
    const savedSettings = localStorage.getItem('ragSettings');
    const origin = DEFAULT_API_BASE_URL;
    let settings = {};
    try {
        settings = savedSettings ? JSON.parse(savedSettings) : {};
    } catch (e) {
        settings = {};
    }

    const storedApiBaseUrl = normalizeBaseUrl(localStorage.getItem('apiBaseUrl'));
    const settingsApiBaseUrl = normalizeBaseUrl(settings.apiBaseUrl);

    let apiBaseUrl = settingsApiBaseUrl || storedApiBaseUrl || origin;
    if (apiBaseUrl.includes("localhost") && !origin.includes("localhost")) {
        apiBaseUrl = origin;
        localStorage.setItem('apiBaseUrl', apiBaseUrl);
        localStorage.setItem('ragSettings', JSON.stringify({ ...settings, apiBaseUrl }));
    }

    if (elements.apiUrlInput) {
        elements.apiUrlInput.value = apiBaseUrl;
    }
    API_BASE_URL = apiBaseUrl || API_BASE_URL;
    updateApiBaseStatus();

    if (settings.topK) {
        elements.topKInput.value = settings.topK;
    }
    if (settings.rerankEnabled) {
        elements.rerankSelect.value = settings.rerankEnabled;
    }
}
