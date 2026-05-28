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

        const refused = data && data.metadata && data.metadata.refused;
        if (refused) {
            const firstSrc = (data.sources && data.sources[0] && data.sources[0].source) || '';
            const candidates = ((data && data.metadata && data.metadata.candidate_sources) || []).filter(Boolean);
            let msg = '';
            if (refused === 'retrieval_error') {
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
            addErrorMessage(msg);
        } else {
            addAssistantMessage(data.answer, data.sources, responseTime);
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
}

// 添加助手消息（带来源）
function addAssistantMessage(answer, sources, responseTime) {
    const message = {
        id: Date.now(),
        content: answer,
        role: 'assistant',
        sources: sources,
        timestamp: new Date().toISOString(),
        responseTime: responseTime
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

    let sourcesHtml = '';
    if (message.sources && message.sources.length > 0) {
        sourcesHtml = `
            <div class="sources-list">
                <p style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                    相关片段（共 ${message.sources.length} 条）：
                </p>
                ${message.sources.map(source => `
                    <div class="source-item">
                        <div><strong>命中文档：</strong>${escapeHtml(source.source || '')}</div>
                        <p style="font-size: 0.875rem; margin-top: 0.25rem; color: var(--text-secondary);">
                            <strong>相关片段：</strong>${escapeHtml(source.text || '')}
                        </p>
                    </div>
                `).join('')}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender-name">AI 助手</span>
                <span class="timestamp">${timestamp} (响应：${message.responseTime}s)</span>
            </div>
            <div class="message-text">
                <p>${escapeHtml(message.content)}</p>
            </div>
            ${sourcesHtml}
        </div>
    `;

    elements.messagesContainer.appendChild(messageDiv);
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

    const createdAt = doc.created_at ? new Date(doc.created_at).toLocaleString('zh-CN') : '-';
    const content = escapeHtml(doc.content || '');

    modal.innerHTML = `
        <div style="padding: 14px 18px; border-bottom: 1px solid #eee; display:flex; justify-content:space-between; align-items:center; gap:10px;">
            <div>
                <div style="font-weight: 600; font-size: 1rem;">${escapeHtml(doc.filename || '文档详情')}</div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 4px;">分块数：${doc.chunk_count || 0} · 上传时间：${createdAt}</div>
            </div>
            <button id="doc-modal-close" style="border:0; background:transparent; font-size:1.25rem; cursor:pointer;">×</button>
        </div>
        <div style="padding: 16px 18px; overflow:auto; white-space: pre-wrap; line-height: 1.6; font-size: 0.92rem; color:#222;">
            ${content || '<span style="color:#999;">（无内容）</span>'}
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

    if (settings.topK) {
        elements.topKInput.value = settings.topK;
    }
    if (settings.rerankEnabled) {
        elements.rerankSelect.value = settings.rerankEnabled;
    }
}
