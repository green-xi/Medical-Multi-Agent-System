/**
 * MedicalAI — 前端主应用
 * 含短期+长期记忆面板
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';


// SECTION 1 — 工具函数


function formatTimeAgo(timestamp) {
  const now = new Date(); const past = new Date(timestamp);
  const diffMs = now - past;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffMins < 1) return '刚刚';
  if (diffMins < 60) return diffMins + '分钟前';
  if (diffHours < 24) return diffHours + '小时前';
  if (diffDays < 7) return diffDays + '天前';
  return past.toLocaleDateString('zh-CN');
}
function getChineseTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
}
function buildDownloadText(chatHistory) {
  let c = 'MedicalAI — 对话记录\n' + '='.repeat(40) + '\n导出时间：' + new Date().toLocaleString('zh-CN') + '\n\n';
  chatHistory.forEach(m => {
    c += '[' + m.timestamp + '] ' + (m.type === 'user' ? '我' : 'MedicalAI') + '：\n' + m.content + '\n';
    if (m.source) c += '来源：' + m.source + '\n';
    c += '\n';
  });
  return c + '\n— 此记录仅供参考，不构成医疗诊断建议 —\n';
}


// SECTION 2 — 长期记忆面板


const PROFILE_LABELS = { age:'年龄', gender:'性别', allergies:'过敏史', conditions:'既往病史', medications:'当前用药' };
const FACT_LABELS = { chief_complaint:'主诉', symptoms:'主要症状', symptom_duration:'持续时长', diagnosis:'诊断建议', advice:'医嘱' };

function MemoryPanel({ memory, onClearMemory, onManualAdd, isLoading }) {
  const [tab, setTab] = useState('profile');
  const [addOpen, setAddOpen] = useState(false);
  const [addKey, setAddKey] = useState('');
  const [addValue, setAddValue] = useState('');
  const [addType, setAddType] = useState('user_profile');

  const profileItems = memory?.user_profile || [];
  const factItems = memory?.medical_facts || [];
  const labelFor = (key, type) => type === 'user_profile' ? (PROFILE_LABELS[key] || key) : (FACT_LABELS[key] || key);

  const handleAdd = () => {
    if (!addKey.trim() || !addValue.trim()) return;
    onManualAdd({ memory_type: addType, key: addKey.trim(), value: addValue.trim(), importance: 7 });
    setAddKey(''); setAddValue(''); setAddOpen(false);
  };

  return (
    <div className="memory-panel glass-effect">
      <div className="memory-panel-header">
        <div className="memory-panel-title">
          <i className="fas fa-brain" /><span>记忆系统</span>
          {isLoading && <span className="mem-loading-dot" />}
        </div>
        <div className="memory-tabs">
          <button className={'mem-tab' + (tab==='profile'?' active':'')} onClick={()=>setTab('profile')}>
            <i className="fas fa-user-circle" /> 患者档案
            {profileItems.length > 0 && <span className="mem-badge">{profileItems.length}</span>}
          </button>
          <button className={'mem-tab' + (tab==='facts'?' active':'')} onClick={()=>setTab('facts')}>
            <i className="fas fa-notes-medical" /> 就诊记录
            {factItems.length > 0 && <span className="mem-badge">{factItems.length}</span>}
          </button>
        </div>
      </div>

      <div className="memory-panel-body">
        {tab === 'profile' && (profileItems.length === 0
          ? <div className="memory-empty"><i className="fas fa-user-slash" /><span>暂无患者档案</span><p>AI 将在对话中自动提取</p></div>
          : <div className="memory-list">{profileItems.map(m=>(
              <div key={m.id} className="memory-item">
                <span className="mem-key">{labelFor(m.key,'user_profile')}</span>
                <span className="mem-value">{m.value}</span>
                <span className={'mem-imp imp-'+(m.importance>=8?'high':m.importance>=5?'mid':'low')}>
                  {'★'.repeat(Math.min(Math.ceil(m.importance/3.3),3))}
                </span>
              </div>
            ))}</div>
        )}
        {tab === 'facts' && (factItems.length === 0
          ? <div className="memory-empty"><i className="fas fa-file-medical-alt" /><span>暂无就诊记录</span><p>每轮对话后自动提取</p></div>
          : <div className="memory-list">{factItems.map(m=>(
              <div key={m.id} className="memory-item">
                <span className="mem-key">{labelFor(m.key,'medical_fact')}</span>
                <span className="mem-value">{m.value}</span>
              </div>
            ))}</div>
        )}
      </div>

      {addOpen && (
        <div className="memory-add-form glass-effect">
          <select value={addType} onChange={e=>setAddType(e.target.value)} className="mem-select">
            <option value="user_profile">患者档案</option>
            <option value="medical_fact">就诊记录</option>
          </select>
          <input className="mem-input" placeholder="字段（如 allergies）" value={addKey} onChange={e=>setAddKey(e.target.value)} />
          <input className="mem-input" placeholder="内容（如 青霉素过敏）" value={addValue} onChange={e=>setAddValue(e.target.value)}
            onKeyDown={e=>{ if(e.key==='Enter') handleAdd(); }} />
          <div className="mem-add-btns">
            <button className="mem-btn-confirm" onClick={handleAdd}><i className="fas fa-check" /> 保存</button>
            <button className="mem-btn-cancel" onClick={()=>setAddOpen(false)}><i className="fas fa-times" /> 取消</button>
          </div>
        </div>
      )}

      <div className="memory-panel-footer">
        <button className="mem-action-btn" onClick={()=>setAddOpen(o=>!o)} title="手动添加记忆条目">
          <i className="fas fa-plus-circle" /><span>添加</span>
        </button>
        <button className="mem-action-btn danger" onClick={onClearMemory} title="清除全部长期记忆">
          <i className="fas fa-trash-alt" /><span>清除记忆</span>
        </button>
      </div>
    </div>
  );
}


// SECTION 3 — 侧边栏


function Sidebar({ sidebarOpen, sessions, currentSessionId, onNewChat, onLoadSession, onDeleteSession, onToggleTheme, theme, memory, onClearMemory, onManualAdd, memoryLoading }) {
  return (
    <aside className={'sidebar glass-effect' + (sidebarOpen ? '' : ' collapsed')}>
      <div className="sidebar-content">
        <div className="sidebar-header">
          <div className="logo-wrapper">
            <div className="logo-animated"><div className="logo-pulse" /><i className="fas fa-heartbeat" /></div>
            <div className="logo-text"><h1>MedicalAI</h1><span className="version">智能诊询 v3.0</span></div>
          </div>
          <button className="new-chat-btn" onClick={onNewChat}><i className="fas fa-plus" /><span>新建对话</span></button>
        </div>

        <MemoryPanel memory={memory} onClearMemory={onClearMemory} onManualAdd={onManualAdd} isLoading={memoryLoading} />

        <div className="chat-history-section">
          <div className="section-header"><span>历史对话</span><div className="section-line" /></div>
          <div className="chat-list">
            {sessions === null ? (
              <div className="chat-list-empty"><div className="loading-spinner" style={{margin:'0 auto 10px'}} />加载中…</div>
            ) : sessions.length === 0 ? (
              <div className="chat-list-empty">暂无历史对话</div>
            ) : sessions.map(session => (
              <div key={session.session_id}
                className={'chat-item' + (currentSessionId===session.session_id?' active':'')}
                onClick={()=>onLoadSession(session.session_id)}>
                <i className="fas fa-message" />
                <div className="chat-item-content">
                  <div className="chat-item-title">{session.preview||'新对话'}</div>
                  <div className="chat-item-time">{formatTimeAgo(session.last_active)}</div>
                </div>
                <button className="chat-item-delete" title="删除对话"
                  onClick={e=>{e.stopPropagation();onDeleteSession(session.session_id);}}>
                  <i className="fas fa-trash" />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="disclaimer-card glass-effect">
            <div className="disclaimer-header"><i className="fas fa-shield-halved" /><span>免责声明</span></div>
            <p className="disclaimer-text">本助手提供的信息仅供健康参考，不替代专业医疗诊断。紧急情况请拨打 <strong>120</strong>。</p>
          </div>
          <button className="theme-btn glass-effect" onClick={onToggleTheme}>
            <i className={'fas '+(theme==='dark'?'fa-sun':'fa-moon')} />
            <span>{theme==='dark'?'亮色模式':'暗色模式'}</span>
          </button>
        </div>
      </div>
    </aside>
  );
}


// SECTION 4 — 欢迎屏


const QUICK_QUESTIONS = [
  { icon:'fa-thermometer', label:'发烧症状', q:'发烧有哪些常见症状？如何判断需要就医？' },
  { icon:'fa-head-side-virus', label:'头痛处理', q:'头痛怎么办？有哪些缓解方法？' },
  { icon:'fa-heart-pulse', label:'高血压', q:'高血压的原因和日常管理方法有哪些？' },
  { icon:'fa-notes-medical', label:'糖尿病', q:'糖尿病患者日常饮食需要注意什么？' },
  { icon:'fa-lungs', label:'咳嗽用药', q:'持续咳嗽应该怎么处理？什么情况需要就医？' },
  { icon:'fa-pills', label:'感冒对策', q:'感冒了有哪些有效的缓解方法？' },
];
const FEATURES = [
  { icon:'fa-brain', label:'AI 驱动', desc:'多智能体协同决策' },
  { icon:'fa-book-medical', label:'医学文献', desc:'RAG 检索增强生成' },
  { icon:'fa-memory', label:'记忆系统', desc:'短期窗口 + 长期档案' },
];

function WelcomeScreen({ onQuickQuestion }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="logo-3d"><i className="fas fa-stethoscope" /><div className="logo-ring" /></div>
        <h1 className="welcome-title">你好，我是 MedicalAI</h1>
        <p className="welcome-subtitle">
          基于大语言模型 · RAG检索增强 · 多智能体架构<br />
          短期记忆 + 长期档案，为您提供专业、连贯的健康咨询
        </p>
        <div className="feature-grid">
          {FEATURES.map(f=>(
            <div key={f.label} className="feature-card glass-effect">
              <div className="feature-icon"><i className={'fas '+f.icon} /></div>
              <div className="feature-text"><strong>{f.label}</strong><span>{f.desc}</span></div>
            </div>
          ))}
        </div>
        <div className="quick-actions">
          <h3><i className="fas fa-bolt" /> 快速提问</h3>
          <div className="quick-buttons">
            {QUICK_QUESTIONS.map(item=>(
              <button key={item.q} className="quick-btn glass-effect" onClick={()=>onQuickQuestion(item.q)}>
                <i className={'fas '+item.icon} /><span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}


// SECTION 4b — 思考过程气泡


const INTENT_CN = {
  symptom_inquiry:       { label:'症状咨询',   icon:'fa-stethoscope',    color:'#7c3aed' },
  medication_inquiry:    { label:'用药询问',   icon:'fa-pills',          color:'#0891b2' },
  report_interpretation: { label:'报告解读',   icon:'fa-file-medical',   color:'#059669' },
  disease_inquiry:       { label:'疾病查询',   icon:'fa-book-medical',   color:'#d97706' },
  treatment_inquiry:     { label:'治疗咨询',   icon:'fa-hand-holding-medical', color:'#dc2626' },
  general_health:        { label:'健康咨询',   icon:'fa-heart-pulse',    color:'#6366f1' },
  chitchat:              { label:'日常交流',   icon:'fa-comments',       color:'#64748b' },
};

function ThinkingBubble({ msg }) {
  const [expanded, setExpanded] = useState(false);
  const { thinking_steps=[], original_question='', query_intent='', rag_think_log=[], tool_trace=[] } = msg.thinking || {};
  const intentMeta = INTENT_CN[query_intent] || { label:'分析中', icon:'fa-brain', color:'#6366f1' };
  const hasRewrite = original_question && original_question !== msg.content;
  const hasRagLog  = rag_think_log && rag_think_log.length > 0;
  const hasSteps   = thinking_steps && thinking_steps.length > 0;
  if (!hasSteps && !hasRagLog) return null;

  return (
    <div className="thinking-bubble-wrap">
      <button className={"thinking-toggle" + (expanded ? " open" : "")} onClick={() => setExpanded(v => !v)}>
        <span className="thinking-toggle-left">
          <span className="thinking-icon-ring" style={{background: intentMeta.color + '22', color: intentMeta.color}}>
            <i className={"fas " + intentMeta.icon} />
          </span>
          <span className="thinking-label">
            <span className="thinking-intent-tag" style={{color: intentMeta.color}}>{intentMeta.label}</span>
            <span className="thinking-subtitle">查看 AI 思考过程</span>
          </span>
        </span>
        <i className={"fas fa-chevron-down thinking-chevron" + (expanded ? " rotated" : "")} />
      </button>

      {expanded && (
        <div className="thinking-body">
          {/* 查询重写 */}
          {hasRewrite && (
            <div className="thinking-section">
              <div className="thinking-section-title"><i className="fas fa-pen-to-square" /> 查询优化</div>
              <div className="thinking-rewrite">
                <span className="rewrite-original"><i className="fas fa-quote-left" /> {original_question}</span>
                <i className="fas fa-arrow-right rewrite-arrow" />
                <span className="rewrite-new">{msg.rewritten}</span>
              </div>
            </div>
          )}

          {/* 意图分析步骤 */}
          {hasSteps && (
            <div className="thinking-section">
              <div className="thinking-section-title"><i className="fas fa-magnifying-glass" /> 意图分析</div>
              <ol className="thinking-steps">
                {thinking_steps.map((step, i) => (
                  <li key={i} className="thinking-step">
                    <span className="step-num">{i + 1}</span>
                    <span className="step-text">{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {/* 执行路径 */}
          {tool_trace && tool_trace.length > 0 && (
            <div className="thinking-section">
              <div className="thinking-section-title"><i className="fas fa-route" /> 执行路径</div>
              <div className="trace-path">
                {tool_trace.filter(t => !t.startsWith('rag_grader:iter')).map((t, i) => (
                  <span key={i} className="trace-node">{t}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// SECTION 5 — 消息气泡


const SOURCE_META = {
  '医学知识库':{ icon:'fa-brain', color:'#7c3aed' },
  '通用医疗知识':{ icon:'fa-book-medical', color:'#0891b2' },
  'Wikipedia 医学资料':{ icon:'fa-globe', color:'#374151' },
  '实时医学搜索':{ icon:'fa-satellite-dish', color:'#059669' },
  '医学知识库（抽取式回退）':{ icon:'fa-database', color:'#6366f1' },
  '系统提示':{ icon:'fa-circle-info', color:'#d97706' },
};

function SourceBadge({ source }) {
  if (!source || source==='系统提示') return null;
  const meta = SOURCE_META[source] || { icon:'fa-database', color:'#6366f1' };
  return <span className="message-source" style={{'--src-color':meta.color}}><i className={'fas '+meta.icon} />{source}</span>;
}

function MessageBubble({ msg }) {
  const [copied, setCopied] = useState(false);
  const copyText = useCallback(()=>{
    navigator.clipboard.writeText(msg.content).then(()=>{setCopied(true);setTimeout(()=>setCopied(false),2000);}).catch(()=>{});
  },[msg.content]);

  if (msg.type==='user') return (
    <div className="message user-message"><div className="message-wrapper">
      <div className="message-content">
        <div className="message-text user-text">{msg.content}</div>
        <span className="message-time">{msg.timestamp}</span>
      </div>
      <div className="message-avatar user-avatar"><i className="fas fa-user" /></div>
    </div></div>
  );

  const hasThinking = msg.thinking && (
    (msg.thinking.thinking_steps && msg.thinking.thinking_steps.length > 0) ||
    (msg.thinking.tool_trace     && msg.thinking.tool_trace.length     > 0)
  );

  return (
    <div className="message bot-message"><div className="message-wrapper">
      <div className="message-avatar bot-avatar"><i className="fas fa-robot" /><span className="avatar-ping" /></div>
      <div className="message-content">
        <div className="bot-name">MedicalAI</div>
        {hasThinking && <ThinkingBubble msg={msg} />}
        <div className="message-text bot-text"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
        <div className="message-footer">
          <span className="message-time">{msg.timestamp}</span>
          <SourceBadge source={msg.source} />
          <button className={'message-action'+(copied?' copied':'')} title={copied?'已复制':'复制回复'} onClick={copyText}>
            <i className={'fas '+(copied?'fa-check':'fa-copy')} />
            {copied && <span className="copy-tip">已复制</span>}
          </button>
        </div>
      </div>
    </div></div>
  );
}


// SECTION 6 — 聊天区域


function ChatArea({ messages, isTyping, progressSteps, showWelcome, onQuickQuestion, chatAreaRef }) {
  return (
    <div className="chat-area" ref={chatAreaRef}>
      {showWelcome && <WelcomeScreen onQuickQuestion={onQuickQuestion} />}
      <div className="messages-container">
        {messages.map((msg,idx)=><MessageBubble key={idx} msg={msg} />)}
      </div>
      {isTyping && (
        <div className="typing-indicator active">
          <div className="typing-bubble glass-effect">
            <div className="message-avatar bot-avatar small"><i className="fas fa-robot" /></div>
            <div className="typing-content">
              {progressSteps.length > 0 ? (
                <div className="typing-progress">
                  {progressSteps.map((step, i) => (
                    <div key={i} className={"typing-step-item" + (i === progressSteps.length - 1 ? " active" : " done")}>
                      <span className="typing-step-icon">
                        {i === progressSteps.length - 1
                          ? <i className="fas fa-spinner fa-spin" style={{fontSize:'11px'}} />
                          : <i className="fas fa-check" style={{fontSize:'10px'}} />}
                      </span>
                      <span className="typing-step-text">{step}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <>
                  <span className="typing-text">MedicalAI 正在思考</span>
                  <div className="typing-dots"><span className="dot" /><span className="dot" /><span className="dot" /></div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


// SECTION 7 — 输入区域


function InputArea({ inputValue, setInputValue, onSend, isTyping, inputRef }) {
  const maxChars = 500;
  const handleKeyDown = e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();onSend();} };
  const handleInput = e=>{
    setInputValue(e.target.value);
    e.target.style.height='auto';
    e.target.style.height=Math.min(e.target.scrollHeight,120)+'px';
  };
  return (
    <div className="input-area"><div className="input-wrapper">
      <div className={'input-container glass-effect'+(isTyping?' disabled':'')}>
        <textarea ref={inputRef} className="message-input"
          placeholder="请描述您的健康问题… (Enter 发送，Shift+Enter 换行)"
          rows={1} value={inputValue} onChange={handleInput} onKeyDown={handleKeyDown}
          maxLength={maxChars} disabled={isTyping} />
        <div className="input-controls">
          <span className={'char-count'+(inputValue.length>maxChars*0.8?' warn':'')}>{inputValue.length}/{maxChars}</span>
          <button className="send-btn" title="发送" onClick={onSend} disabled={!inputValue.trim()||isTyping}>
            {isTyping?<i className="fas fa-spinner fa-spin" />:<i className="fas fa-paper-plane" />}
          </button>
        </div>
      </div>
      <div className="input-info">
        <i className="fas fa-circle-exclamation" />
        <span>AI回答仅供健康参考，不构成医疗诊断。紧急情况请立即就医或拨打 <strong>120</strong>。</span>
      </div>
    </div></div>
  );
}


// SECTION 8 — 根组件


const API_BASE = '/api/v1';

// ── 统一请求封装：自动附带 X-Session-ID Header ────────────────────────────────
function apiHeaders(sessionId, extra = {}) {
  const h = { ...extra };
  if (sessionId) h['X-Session-ID'] = sessionId;
  return h;
}
function apiFetch(url, options = {}, sessionId = null) {
  const { headers: extraHeaders, ...rest } = options;
  return fetch(url, {
    ...rest,
    headers: apiHeaders(sessionId, extraHeaders),
  });
}

const TOAST_COLORS = {
  success:'linear-gradient(135deg,#10b981,#059669)',
  error:'linear-gradient(135deg,#ef4444,#dc2626)',
  info:'linear-gradient(135deg,#3b82f6,#2563eb)',
  warn:'linear-gradient(135deg,#f59e0b,#d97706)',
};
const TOAST_ICONS = { success:'fa-circle-check', error:'fa-circle-xmark', info:'fa-circle-info', warn:'fa-triangle-exclamation' };

function useIsMobile(bp=768) {
  const [v,setV]=useState(()=>window.innerWidth<=bp);
  useEffect(()=>{const h=()=>setV(window.innerWidth<=bp);window.addEventListener('resize',h);return()=>window.removeEventListener('resize',h);},[bp]);
  return v;
}

export default function App() {
  const [theme,setTheme]=useState(()=>localStorage.getItem('theme')||'light');
  const isMobile=useIsMobile();
  const [sidebarOpen,setSidebarOpen]=useState(()=>{
    if(window.innerWidth<=768)return false;
    return localStorage.getItem('sidebarOpen')!=='false';
  });
  const [sessions,setSessions]=useState(null);
  const [currentSessionId,setCurrentSessionId]=useState(null);
  const [messages,setMessages]=useState([]);
  const [chatHistory,setChatHistory]=useState([]);
  const [showWelcome,setShowWelcome]=useState(true);
  const [isTyping,setIsTyping]=useState(false);
  const [progressSteps,setProgressSteps]=useState([]); // 流式进度步骤
  const [inputValue,setInputValue]=useState('');
  const [toast,setToast]=useState({show:false,message:'',type:'success'});
  const [memory,setMemory]=useState(null);
  const [memoryLoading,setMemoryLoading]=useState(false);

  const chatAreaRef=useRef(null);
  const inputRef=useRef(null);
  const toastTimerRef=useRef(null);

  useEffect(()=>{document.documentElement.setAttribute('data-theme',theme);localStorage.setItem('theme',theme);},[theme]);
  const toggleTheme=()=>setTheme(t=>t==='light'?'dark':'light');
  const toggleSidebar=()=>setSidebarOpen(prev=>{if(!isMobile)localStorage.setItem('sidebarOpen',!prev);return !prev;});
  const closeSidebar=()=>setSidebarOpen(false);

  const showToast=useCallback((message,type='success')=>{
    if(toastTimerRef.current)clearTimeout(toastTimerRef.current);
    setToast({show:true,message,type});
    toastTimerRef.current=setTimeout(()=>setToast(t=>({...t,show:false})),3000);
  },[]);

  const scrollToBottom=useCallback(()=>{
    if(chatAreaRef.current)chatAreaRef.current.scrollTo({top:chatAreaRef.current.scrollHeight,behavior:'smooth'});
  },[]);
  useEffect(()=>{scrollToBottom();},[messages,isTyping,scrollToBottom]);

  // ── 记忆操作 ──────────────────────────────────────────────────
  const loadMemory=useCallback(async(sessionId)=>{
    setMemoryLoading(true);
    try{
      const sid=sessionId||currentSessionId;
      const res=await apiFetch(API_BASE+'/memory',{},sid);
      const data=await res.json();
      if(data.success)setMemory({user_profile:data.user_profile,medical_facts:data.medical_facts});
    }catch{}finally{setMemoryLoading(false);}
  },[currentSessionId]);

  const clearMemory=useCallback(async()=>{
    if(!window.confirm('确定要清除全部长期记忆（患者档案+就诊记录）吗？\n对话记录不受影响。'))return;
    try{
      const res=await apiFetch(API_BASE+'/memory',{method:'DELETE'},currentSessionId);
      if(res.ok){setMemory({user_profile:[],medical_facts:[]});showToast('长期记忆已清除','success');}
    }catch{showToast('清除失败','error');}
  },[showToast,currentSessionId]);

  const manualAddMemory=useCallback(async(body)=>{
    try{
      const res=await apiFetch(API_BASE+'/memory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)},currentSessionId);
      if(res.ok){await loadMemory(currentSessionId);showToast('记忆已保存','success');}
    }catch{showToast('保存失败','error');}
  },[loadMemory,showToast,currentSessionId]);

  // ── 会话操作 ──────────────────────────────────────────────────
  const loadSessions=useCallback(async()=>{
    try{const res=await apiFetch(API_BASE+'/sessions');const data=await res.json();if(data.success&&data.sessions)setSessions(data.sessions);}
    catch{setSessions([]);}
  },[]);

  useEffect(()=>{
    loadSessions(); loadMemory();
    (async()=>{
      try{
        const res=await apiFetch(API_BASE+'/history');const data=await res.json();
        if(data.success&&data.messages&&data.messages.length>0){
          const msgs=data.messages.map(m=>({type:m.role==='user'?'user':'assistant',content:m.content,timestamp:m.timestamp||'',source:m.source||null}));
          setMessages(msgs);setChatHistory(msgs.map(m=>({...m})));setShowWelcome(false);
        }
      }catch{}
    })();
  },[loadSessions,loadMemory]);

  const loadSession=useCallback(async(sessionId)=>{
    try{
      const res=await apiFetch(API_BASE+'/session/'+sessionId,{},sessionId);const data=await res.json();
      if(data.success){
        setCurrentSessionId(sessionId);
        const msgs=data.messages.map(m=>({type:m.role==='user'?'user':'assistant',content:m.content,timestamp:m.timestamp||'',source:m.source||null}));
        setMessages(msgs);setChatHistory(msgs.map(m=>({...m})));setShowWelcome(false);
        await loadMemory(sessionId);
        if(isMobile)closeSidebar();
        showToast('对话加载成功','success');
      }
    }catch{showToast('加载失败，请重试','error');}
  },[showToast,isMobile,loadMemory]);

  const deleteSession=useCallback(async(sessionId)=>{
    if(!window.confirm('确定要删除这条对话记录吗？'))return;
    try{const res=await apiFetch(API_BASE+'/session/'+sessionId,{method:'DELETE'},sessionId);if(res.ok){await loadSessions();showToast('对话已删除','success');}}
    catch{showToast('删除失败','error');}
  },[loadSessions,showToast]);

  const createNewChat=useCallback(async()=>{
    try{
      const res=await apiFetch(API_BASE+'/new-chat',{method:'POST'},currentSessionId);
      if(res.ok){
        const data=await res.json();
        const newSessionId=data.session_id||null;
        setMessages([]);setChatHistory([]);setCurrentSessionId(newSessionId);setShowWelcome(true);
        await loadSessions();await loadMemory(newSessionId);
        if(isMobile)closeSidebar();
        showToast('新对话已创建','success');
      }
    }catch{showToast('创建失败','error');}
  },[loadSessions,loadMemory,showToast,isMobile,currentSessionId]);

  const clearChat=useCallback(async()=>{
    if(!window.confirm('确定要清空当前对话吗？'))return;
    try{const res=await apiFetch(API_BASE+'/clear',{method:'POST'},currentSessionId);if(res.ok){setMessages([]);setChatHistory([]);setShowWelcome(true);showToast('对话已清空','success');}}
    catch{showToast('清空失败','error');}
  },[showToast,currentSessionId]);

  const downloadChat=useCallback(()=>{
    if(chatHistory.length===0){showToast('暂无消息可下载','warn');return;}
    const blob=new Blob([buildDownloadText(chatHistory)],{type:'text/plain;charset=utf-8'});
    const url=URL.createObjectURL(blob);const a=document.createElement('a');
    a.href=url;a.download='MedicalAI_对话记录_'+new Date().toLocaleDateString('zh-CN').replace(/\//g,'-')+'.txt';
    a.click();URL.revokeObjectURL(url);showToast('对话记录已下载','success');
  },[chatHistory,showToast]);

  const sendMessage=useCallback(async(overrideText)=>{
    const message=(overrideText!==undefined?overrideText:inputValue).trim();
    if(!message||isTyping)return;
    setShowWelcome(false);
    const time=getChineseTime();
    const userMsg={type:'user',content:message,timestamp:time,source:null};
    setMessages(prev=>[...prev,userMsg]);setChatHistory(prev=>[...prev,userMsg]);
    setInputValue('');if(inputRef.current)inputRef.current.style.height='auto';
    setIsTyping(true);
    setProgressSteps([]); // 重置进度步骤
    try{
      // ── 使用 SSE 流式接口，实时展示思考进度 ──────────────────────────────
      const headers={'Content-Type':'application/json'};
      if(currentSessionId)headers['X-Session-ID']=currentSessionId;
      const res=await fetch(API_BASE+'/chat/stream',{
        method:'POST',
        headers,
        body:JSON.stringify({message}),
      });
      if(!res.ok||!res.body){
        // 降级：流式不可用时回退到普通 /chat
        const fallback=await apiFetch(API_BASE+'/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message})},currentSessionId);
        const data=await fallback.json();
        if(data.success){
          if(data.session_id&&!currentSessionId)setCurrentSessionId(data.session_id);
          const botMsg={type:'assistant',content:data.response,timestamp:data.timestamp||time,source:data.source||null,
            rewritten:data.rewritten_question||data.question||'',
            thinking:{thinking_steps:data.thinking_steps||[],original_question:data.original_question||'',query_intent:data.query_intent||'',rag_think_log:data.rag_think_log||[],tool_trace:data.tool_trace||[]}};
          setMessages(prev=>[...prev,botMsg]);setChatHistory(prev=>[...prev,botMsg]);
        }
        return;
      }
      // ── 解析 SSE 数据流 ─────────────────────────────────────────────────
      const reader=res.body.getReader();
      const decoder=new TextDecoder();
      let buffer='';
      let pendingThinking=null;

      // 解析单个 SSE 事件块
      const parseChunk=(chunk)=>{
        if(!chunk.trim())return;
        let evtType='message',dataStr='';
        for(const line of chunk.split('\n')){
          if(line.startsWith('event:'))evtType=line.slice(6).trim();
          else if(line.startsWith('data:'))dataStr+=line.slice(5).trim();
        }
        console.debug('[SSE] event='+evtType+' dataLen='+dataStr.length);
        try{
          const payload=JSON.parse(dataStr);
          if(evtType==='progress'){
            setProgressSteps(prev=>[...prev,payload.step]);
          }else if(evtType==='thinking'){
            pendingThinking=payload;
          }else if(evtType==='done'){
            const data=payload;
            if(data.session_id&&!currentSessionId)setCurrentSessionId(data.session_id);
            const botMsg={
              type:'assistant',
              content:data.response,
              timestamp:data.timestamp||time,
              source:data.source||null,
              rewritten:data.rewritten_question||data.question||'',
              thinking:{
                thinking_steps:  (pendingThinking?.thinking_steps)  || data.thinking_steps  || [],
                original_question:(pendingThinking?.original_question)||data.original_question||'',
                query_intent:    (pendingThinking?.query_intent)     || data.query_intent    || '',
                rag_think_log:   data.rag_think_log  || [],
                tool_trace:      data.tool_trace     || [],
              }
            };
            setMessages(prev=>[...prev,botMsg]);setChatHistory(prev=>[...prev,botMsg]);
            loadSessions();
            const sid=data.session_id||currentSessionId;
            setTimeout(()=>loadMemory(sid),1500);
          }else if(evtType==='error'){
            setMessages(prev=>[...prev,{type:'assistant',content:'抱歉，处理您的问题时出现错误，请稍后重试。',timestamp:time,source:null}]);
            showToast('回复失败，请重试','error');
          }
        }catch(e){
          console.warn('[SSE] JSON解析失败 event='+evtType,e,'raw='+dataStr.slice(0,200));
        }
      };

      while(true){
        const {done,value}=await reader.read();
        if(value)buffer+=decoder.decode(value,{stream:true});
        // 流结束时处理 buffer 中残留的最后一帧（done 事件 JSON 较大，可能不含结尾 \n\n）
        if(done){
          if(buffer.trim())parseChunk(buffer);
          break;
        }
        const frames=buffer.split('\n\n');
        buffer=frames.pop()||'';
        for(const frame of frames)parseChunk(frame);
      }
    }catch{
      setMessages(prev=>[...prev,{type:'assistant',content:'网络连接失败，请检查网络后重试。',timestamp:time,source:null}]);
      showToast('网络连接错误','error');
    }finally{setIsTyping(false);setProgressSteps([]);}
  },[inputValue,isTyping,loadSessions,loadMemory,showToast,currentSessionId]);

  const handleQuickQuestion=useCallback(q=>{setTimeout(()=>sendMessage(q),100);},[sendMessage]);

  return (
    <>
      <div className="animated-background">
        <div className="gradient-overlay" />
        <div className="floating-circles">
          <div className="circle circle-1" /><div className="circle circle-2" /><div className="circle circle-3" />
        </div>
      </div>
      <div className="app-container">
        <button className="sidebar-toggle-btn" onClick={toggleSidebar} title="展开/收起侧边栏"><i className="fas fa-bars" /></button>
        {isMobile&&sidebarOpen&&<div className="sidebar-backdrop" onClick={closeSidebar} />}
        <Sidebar sidebarOpen={sidebarOpen} sessions={sessions} currentSessionId={currentSessionId}
          onNewChat={createNewChat} onLoadSession={loadSession} onDeleteSession={deleteSession}
          onToggleTheme={toggleTheme} theme={theme}
          memory={memory} onClearMemory={clearMemory} onManualAdd={manualAddMemory} memoryLoading={memoryLoading} />

        <main className={'main-content'+(sidebarOpen?' sidebar-open':'')}>
          <header className="app-header glass-header">
            <div className="header-content">
              <div className="header-title-group">
                <div className="header-icon"><i className="fas fa-hospital-user" /></div>
                <div>
                  <h2 className="gradient-text">MedicalAI</h2>
                  <p className="header-subtitle">智能健康咨询 · 短期+长期记忆 · 多智能体架构</p>
                </div>
              </div>
              <div className="status-indicator">
                <div className="status-ring"><span className="ring-pulse" /></div>
                <span>服务就绪</span>
              </div>
            </div>
            <div className="header-actions">
              <button className="action-btn" title="清空对话" onClick={clearChat}><i className="fas fa-trash-can" /></button>
              <button className="action-btn" title="刷新记忆面板" onClick={loadMemory}><i className="fas fa-brain" /></button>
              <button className="action-btn" title="下载记录" onClick={downloadChat}><i className="fas fa-download" /></button>
              <button className="action-btn" title="切换主题" onClick={toggleTheme}><i className={'fas '+(theme==='dark'?'fa-sun':'fa-moon')} /></button>
            </div>
          </header>
          <ChatArea messages={messages} isTyping={isTyping} progressSteps={progressSteps} showWelcome={showWelcome} onQuickQuestion={handleQuickQuestion} chatAreaRef={chatAreaRef} />
          <InputArea inputValue={inputValue} setInputValue={setInputValue} onSend={()=>sendMessage()} isTyping={isTyping} inputRef={inputRef} />
        </main>
      </div>
      <div className={'toast'+(toast.show?' show':'')} style={{background:TOAST_COLORS[toast.type]}}>
        <i className={'fas '+TOAST_ICONS[toast.type]} /><span>{toast.message}</span>
      </div>
    </>
  );
}
