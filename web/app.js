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

        // 添加助手消息
        addAssistantMessage(data.answer, data.sources, responseTime);

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
                    来源 (共 ${message.sources.length} 个):
                </p>
                ${message.sources.map(source => `
                    <div class="source-item">
                        <strong>${escapeHtml(source.source)}</strong>
                        <span class="source-score"> (相似度：${(source.score * 100).toFixed(1)}%)</span>
                        <p style="font-size: 0.875rem; margin-top: 0.25rem; color: var(--text-secondary);">
                            ${escapeHtml(source.text)}
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

    elements.documentsList.innerHTML = state.documents.map(doc => `
        <div class="document-item">
            <div class="document-icon">
                <i class="fas fa-file-alt"></i>
            </div>
            <div class="document-info">
                <div class="document-name">${escapeHtml(doc.filename)}</div>
                <div class="document-meta">
                    上传时间：${new Date(doc.created_at).toLocaleString('zh-CN')}
                </div>
            </div>
            <div class="document-actions">
                <button class="btn btn-secondary" onclick="viewDocument('${doc.filename}')">
                    <i class="fas fa-eye"></i>
                    查看
                </button>
                <button class="btn btn-danger" onclick="deleteDocument('${doc.filename}')">
                    <i class="fas fa-trash"></i>
                    删除
                </button>
            </div>
        </div>
    `).join('');
}

// 显示上传对话框
function showUploadDialog() {
    const fileName = prompt('请输入文档文件名:');
    if (!fileName) return;

    const fileContent = prompt('请输入文档内容:');
    if (!fileContent) return;

    uploadDocument(fileName, fileContent);
}

// 上传文档
async function uploadDocument(filename, content) {
    setLoading(true);

    try {
        const response = await fetch(`${resolveApiBaseUrl()}/documents`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: filename,
                content: content,
                metadata: {
                    department: 'general'
                }
            })
        });

        if (response.ok) {
            alert('文档上传成功！');
            loadDocuments();
            updateStats();
        } else {
            throw new Error('上传失败');
        }
    } catch (error) {
        console.error('Error uploading document:', error);
        alert('上传失败，请重试');
    } finally {
        setLoading(false);
    }
}

// 查看文档（占位）
function viewDocument(filename) {
    alert(`查看文档：${filename}`);
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
