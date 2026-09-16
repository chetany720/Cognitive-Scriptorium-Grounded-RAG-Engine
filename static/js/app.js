/**
 * KORTEX — Client Application Controller
 * Handles corpus telemetry, hybrid retrieval queries, grounded citation linking,
 * file uploads, and evidence inspection.
 */

document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------------
  // Application State
  // -------------------------------------------------------------
  const state = {
    documents: [],
    activeCount: 0,
    totalChunks: 0,
    vocabSize: 0,
    currentRetrievedChunks: [],
    settings: {
      provider: 'builtin',
      model: 'auto',
      top_k: 4,
      similarity_threshold: 0.10,
      temperature: 0.2,
      has_api_key: false
    },
    isQuerying: false
  };

  // -------------------------------------------------------------
  // DOM Element References
  // -------------------------------------------------------------
  const statDocsCount = document.getElementById('stat-docs-count');
  const statChunksCount = document.getElementById('stat-chunks-count');
  const statVocabCount = document.getElementById('stat-vocab-count');
  const statEngineName = document.getElementById('stat-engine-name');

  const corpusShelf = document.getElementById('corpus-shelf');
  const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
  const documentList = document.getElementById('document-list');
  const emptyShelfState = document.getElementById('empty-shelf-state');
  const activeRatioLabel = document.getElementById('active-ratio-label');
  const btnResetSample = document.getElementById('btn-reset-sample');

  const ingestDropzone = document.getElementById('ingest-dropzone');
  const fileInput = document.getElementById('file-input');
  const btnBrowseFile = document.getElementById('btn-browse-file');
  const uploadProgress = document.getElementById('upload-progress');
  const uploadProgressFill = document.getElementById('upload-progress-fill');

  const dialogueScrollArea = document.getElementById('dialogue-scroll-area');
  const scriptoriumWelcome = document.getElementById('scriptorium-welcome');
  const conversationThread = document.getElementById('conversation-thread');
  const synthesisLoader = document.getElementById('synthesis-loader');

  const queryForm = document.getElementById('query-form');
  const queryInput = document.getElementById('query-input');
  const btnSubmitQuery = document.getElementById('btn-submit-query');
  const dockActiveDocsText = document.getElementById('dock-active-docs-text');

  const evidenceInspector = document.getElementById('evidence-inspector');
  const btnToggleEvidence = document.getElementById('btn-toggle-evidence');
  const btnCloseEvidence = document.getElementById('btn-close-evidence');
  const evidenceEmptyState = document.getElementById('evidence-empty-state');
  const evidenceChunksFeed = document.getElementById('evidence-chunks-feed');
  const evidenceCountBadge = document.getElementById('evidence-count-badge');

  const settingsModal = document.getElementById('settings-modal');
  const btnOpenSettings = document.getElementById('btn-open-settings');
  const btnCloseSettings = document.getElementById('btn-close-settings');
  const settingsForm = document.getElementById('settings-form');
  const settingProvider = document.getElementById('setting-provider');
  const settingApiKey = document.getElementById('setting-api-key');
  const btnToggleKeyVisibility = document.getElementById('btn-toggle-key-visibility');
  const settingModel = document.getElementById('setting-model');
  const settingTopK = document.getElementById('setting-topk');
  const valTopKDisplay = document.getElementById('val-topk-display');
  const settingThreshold = document.getElementById('setting-threshold');
  const valThresholdDisplay = document.getElementById('val-threshold-display');
  const settingTemp = document.getElementById('setting-temp');
  const valTempDisplay = document.getElementById('val-temp-display');
  const settingsStatusMsg = document.getElementById('settings-status-msg');

  const paramTopkDisplay = document.getElementById('param-topk-display');
  const paramThresholdDisplay = document.getElementById('param-threshold-display');
  const toastContainer = document.getElementById('toast-container');

  // -------------------------------------------------------------
  // Initial Boot & Telemetry Fetch
  // -------------------------------------------------------------
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const json = await res.json();
      if (json.success) {
        updateTelemetry(json.data);
      }
    } catch (err) {
      console.error('Failed to load system status:', err);
      showToast('Unable to connect to KORTEX server.', 'error');
    }
  }

  function updateTelemetry(data) {
    state.documents = data.documents || [];
    state.totalChunks = data.total_chunks || 0;
    state.vocabSize = data.vocabulary_features || 0;
    state.activeCount = data.active_documents || 0;
    if (data.llm_settings) {
      state.settings = { ...state.settings, ...data.llm_settings };
    }

    // Update Header
    statDocsCount.textContent = `${state.activeCount}/${state.documents.length} Docs`;
    statChunksCount.textContent = `${state.totalChunks} Chunks`;
    statVocabCount.textContent = `${state.vocabSize} Features`;
    
    const engineLabel = {
      'builtin': 'KORTEX Core',
      'gemini': 'Gemini 1.5',
      'openai': 'GPT-4o Mini',
      'groq': 'Groq Llama'
    }[state.settings.provider] || 'KORTEX Core';
    statEngineName.textContent = engineLabel;

    // Update Shelf footer
    paramTopkDisplay.textContent = `${state.settings.top_k} chunks`;
    paramThresholdDisplay.textContent = Number(state.settings.similarity_threshold).toFixed(2);
    activeRatioLabel.textContent = `${state.activeCount}/${state.documents.length} Active`;
    dockActiveDocsText.textContent = `Ready · ${state.activeCount} Documents Active`;

    renderDocumentLedger();
  }

  // -------------------------------------------------------------
  // Document Shelf Rendering
  // -------------------------------------------------------------
  function renderDocumentLedger() {
    documentList.innerHTML = '';

    if (state.documents.length === 0) {
      emptyShelfState.style.display = 'block';
      documentList.appendChild(emptyShelfState);
      return;
    }

    emptyShelfState.style.display = 'none';

    state.documents.forEach((doc) => {
      const card = document.createElement('div');
      card.className = `doc-item-card ${doc.active ? '' : 'inactive'}`;
      card.id = `doc-card-${doc.doc_id}`;

      const ext = (doc.filename.split('.').pop() || 'txt').toLowerCase();

      card.innerHTML = `
        <div class="doc-card-top">
          <span class="doc-format-tag ${ext}">${ext}</span>
          <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
          <div class="doc-controls">
            <label class="toggle-switch" title="Toggle active participation in retrieval">
              <input type="checkbox" ${doc.active ? 'checked' : ''} data-doc-id="${doc.doc_id}">
              <span class="toggle-slider"></span>
            </label>
            <button class="btn-doc-delete" data-doc-id="${doc.doc_id}" title="Remove document from index">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
          </div>
        </div>
        <div class="doc-card-meta">
          <span>${doc.chunks_count} chunks</span>
          <span>·</span>
          <span>${doc.total_words.toLocaleString()} words</span>
        </div>
      `;

      // Event listener for toggle switch
      const toggleCheckbox = card.querySelector('input[type="checkbox"]');
      toggleCheckbox.addEventListener('change', async (e) => {
        const docId = e.target.getAttribute('data-doc-id');
        const isActive = e.target.checked;
        await toggleDocumentActive(docId, isActive);
      });

      // Event listener for delete button
      const deleteBtn = card.querySelector('.btn-doc-delete');
      deleteBtn.addEventListener('click', async () => {
        const docId = deleteBtn.getAttribute('data-doc-id');
        if (confirm(`Purge "${doc.filename}" from the research corpus?`)) {
          await deleteDocument(docId);
        }
      });

      documentList.appendChild(card);
    });
  }

  async function toggleDocumentActive(docId, active) {
    try {
      const res = await fetch(`/api/documents/${docId}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active })
      });
      const json = await res.json();
      if (json.success) {
        updateTelemetry(json.stats);
        showToast('Corpus index recalibrated.');
      }
    } catch (err) {
      console.error(err);
      showToast('Error toggling document.', 'error');
    }
  }

  async function deleteDocument(docId) {
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
      const json = await res.json();
      if (json.success) {
        updateTelemetry(json.stats);
        showToast('Document removed from vector store.');
      }
    } catch (err) {
      console.error(err);
      showToast('Failed to delete document.', 'error');
    }
  }

  btnResetSample.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/documents/reset-sample', { method: 'POST' });
      const json = await res.json();
      if (json.success) {
        updateTelemetry(json.stats);
        showToast('Curated research corpus restored.');
      }
    } catch (err) {
      console.error(err);
      showToast('Failed to reset sample papers.', 'error');
    }
  });

  // -------------------------------------------------------------
  // File Ingestion & Drag & Drop
  // -------------------------------------------------------------
  btnBrowseFile.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  ingestDropzone.addEventListener('click', () => {
    fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    ingestDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      ingestDropzone.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    ingestDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      ingestDropzone.classList.remove('drag-over');
    }, false);
  });

  ingestDropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    uploadProgress.style.display = 'block';
    uploadProgressFill.style.width = '40%';

    try {
      uploadProgressFill.style.width = '70%';
      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData
      });
      const json = await res.json();

      uploadProgressFill.style.width = '100%';
      setTimeout(() => {
        uploadProgress.style.display = 'none';
        uploadProgressFill.style.width = '0%';
      }, 500);

      if (json.success) {
        updateTelemetry(json.stats);
        showToast(`Indexed "${file.name}" into ${json.document.chunks_count} chunks.`);
      } else {
        showToast(json.error || 'Upload failed.', 'error');
      }
    } catch (err) {
      console.error(err);
      uploadProgress.style.display = 'none';
      showToast('Error uploading file.', 'error');
    } finally {
      fileInput.value = '';
    }
  }

  // -------------------------------------------------------------
  // Dialogue & Grounded Synthesis Stream
  // -------------------------------------------------------------
  queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + 'px';
  });

  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      queryForm.dispatchEvent(new Event('submit'));
    }
  });

  // Suggestion chips
  document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const queryText = chip.getAttribute('data-query');
      if (queryText) {
        queryInput.value = queryText;
        queryForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query || state.isQuerying) return;

    if (state.activeCount === 0) {
      showToast('No documents active in the corpus shelf. Enable at least one document.', 'error');
      return;
    }

    // Hide welcome banner if visible
    if (scriptoriumWelcome.style.display !== 'none') {
      scriptoriumWelcome.style.display = 'none';
    }

    state.isQuerying = true;
    btnSubmitQuery.disabled = true;

    // Append user message
    appendUserMessage(query);
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Show loading skeleton
    synthesisLoader.style.display = 'block';
    dialogueScrollArea.scrollTop = dialogueScrollArea.scrollHeight;

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: state.settings.top_k,
          threshold: state.settings.similarity_threshold
        })
      });

      const json = await res.json();
      synthesisLoader.style.display = 'none';

      if (json.success) {
        appendAssistantMessage(json);
        updateEvidenceDrawer(json.retrieved_chunks);
      } else {
        appendErrorMessage(json.error || 'Synthesis error occurred.');
      }
    } catch (err) {
      console.error(err);
      synthesisLoader.style.display = 'none';
      appendErrorMessage('Failed to connect to synthesis engine.');
    } finally {
      state.isQuerying = false;
      btnSubmitQuery.disabled = false;
      dialogueScrollArea.scrollTop = dialogueScrollArea.scrollHeight;
    }
  });

  function appendUserMessage(text) {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const row = document.createElement('div');
    row.className = 'msg-row user';
    row.innerHTML = `
      <div class="msg-header">
        <span class="msg-role-tag user">Research Inquiry</span>
        <span>${timeStr}</span>
      </div>
      <div class="user-msg-content">${escapeHtml(text)}</div>
    `;
    conversationThread.appendChild(row);
  }

  function appendAssistantMessage(data) {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const row = document.createElement('div');
    row.className = 'msg-row assistant';

    const metrics = data.metrics || {};
    const topScore = Math.round((metrics.top_relevance_score || 0) * 100);
    const isGrounded = data.grounded && data.retrieved_chunks && data.retrieved_chunks.length > 0;

    // Format markdown text and link citations
    const formattedHtml = renderMarkdownWithCitations(data.answer, data.retrieved_chunks);

    row.innerHTML = `
      <div class="msg-header">
        <div class="msg-role-tag assistant">
          <span>${escapeHtml(data.engine || 'KORTEX Synthesis')}</span>
          <span class="grounded-badge ${isGrounded ? '' : 'unverified'}">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              ${isGrounded 
                ? '<polyline points="20 6 9 17 4 12"></polyline>' 
                : '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
              }
            </svg>
            ${isGrounded ? 'VERIFIED GROUNDING' : 'UNGROUNDED'}
          </span>
        </div>
        <span>${timeStr}</span>
      </div>
      <div class="assistant-msg-content">
        ${data.warning ? `<div style="color:var(--rose-danger);font-size:0.8rem;margin-bottom:10px;">⚠️ ${escapeHtml(data.warning)}</div>` : ''}
        <div class="synthesis-text">${formattedHtml}</div>
        <div class="msg-telemetry-bar">
          <span class="telemetry-item">Retrieval: <strong>${metrics.retrieval_latency_ms || 0}ms</strong></span>
          <span class="telemetry-item">Synthesis: <strong>${metrics.synthesis_latency_ms || 0}ms</strong></span>
          <span class="telemetry-item">Top Match: <strong>${topScore}%</strong></span>
          <span class="telemetry-item">Chunks: <strong>${metrics.chunks_matched || 0}</strong></span>
          <button class="btn-inspect-link" data-action="inspect">
            Inspect Evidence (${metrics.chunks_matched || 0}) →
          </button>
        </div>
      </div>
    `;

    // Attach click listener for citation badges inside this message
    row.querySelectorAll('.citation-pill').forEach(pill => {
      pill.addEventListener('click', (e) => {
        e.preventDefault();
        const cid = parseInt(pill.getAttribute('data-citation'), 10);
        focusCitationInEvidenceDrawer(cid);
      });
    });

    // Inspect link
    const inspectBtn = row.querySelector('.btn-inspect-link');
    if (inspectBtn) {
      inspectBtn.addEventListener('click', () => {
        openEvidenceDrawer();
      });
    }

    conversationThread.appendChild(row);
  }

  function appendErrorMessage(errorText) {
    const row = document.createElement('div');
    row.className = 'msg-row assistant';
    row.innerHTML = `
      <div class="assistant-msg-content" style="border-left-color: var(--rose-danger);">
        <p style="color: var(--rose-danger); font-size:0.9rem;">⚠️ ${escapeHtml(errorText)}</p>
      </div>
    `;
    conversationThread.appendChild(row);
  }

  // -------------------------------------------------------------
  // Markdown & Citation Parsing
  // -------------------------------------------------------------
  function renderMarkdownWithCitations(rawText, chunks) {
    if (!rawText) return '';

    // First, convert citation markers like [1], [2], [1][2] into interactive buttons
    let parsed = rawText.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<button class="citation-pill" data-citation="${p1}" title="Click to inspect Source Context [${p1}]">[${p1}]</button>`;
    });

    // Basic markdown parsing
    // Headers ###
    parsed = parsed.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    parsed = parsed.replace(/^## (.*$)/gim, '<h3>$1</h3>');

    // Blockquotes
    parsed = parsed.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // Bold & Italics
    parsed = parsed.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    parsed = parsed.replace(/\*(.*?)\*/gim, '<em>$1</em>');

    // Code blocks & inline code
    parsed = parsed.replace(/`([^`]+)`/gim, '<code>$1</code>');

    // Bullet points
    parsed = parsed.replace(/^- (.*$)/gim, '<li>$1</li>');
    parsed = parsed.replace(/^• (.*$)/gim, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    parsed = parsed.replace(/(<li>.*<\/li>(\s*<br\s*\/?>)?)+/gim, (m) => `<ul>${m}</ul>`);

    // Paragraphs for line breaks
    parsed = parsed.replace(/\n\n+/g, '</p><p>');
    parsed = `<p>${parsed}</p>`;

    // Clean up empty paragraphs
    parsed = parsed.replace(/<p>\s*<\/p>/g, '');
    parsed = parsed.replace(/<p>\s*(<h3>.*?<\/h3>)\s*<\/p>/g, '$1');
    parsed = parsed.replace(/<p>\s*(<blockquote>.*?<\/blockquote>)\s*<\/p>/g, '$1');
    parsed = parsed.replace(/<p>\s*(<ul>.*?<\/ul>)\s*<\/p>/g, '$1');

    return parsed;
  }

  // -------------------------------------------------------------
  // Evidence & Provenance Inspector
  // -------------------------------------------------------------
  function updateEvidenceDrawer(chunks) {
    state.currentRetrievedChunks = chunks || [];
    evidenceCountBadge.textContent = state.currentRetrievedChunks.length;

    evidenceChunksFeed.innerHTML = '';

    if (state.currentRetrievedChunks.length === 0) {
      evidenceEmptyState.style.display = 'block';
      return;
    }

    evidenceEmptyState.style.display = 'none';

    state.currentRetrievedChunks.forEach((item) => {
      const chunk = item.chunk;
      const cid = item.citation_id;
      const scorePct = Math.round(item.score * 100);

      const card = document.createElement('div');
      card.className = 'chunk-card';
      card.id = `evidence-chunk-${cid}`;

      // Highlight matched terms if any
      let excerptHtml = escapeHtml(chunk.content);
      if (item.matched_keywords && item.matched_keywords.length > 0) {
        const regex = new RegExp(`\\b(${item.matched_keywords.join('|')})\\b`, 'gi');
        excerptHtml = excerptHtml.replace(regex, '<mark>$1</mark>');
      }

      card.innerHTML = `
        <div class="chunk-card-header">
          <div class="chunk-source-cluster">
            <span class="chunk-id-tag">SOURCE [${cid}]</span>
            <span class="chunk-doc-filename" title="${escapeHtml(chunk.doc_name)}">${escapeHtml(chunk.doc_name)}</span>
          </div>
          <div class="chunk-score-meter">
            <span class="score-badge">${scorePct}% Match</span>
            <div class="score-progress-bar">
              <div class="score-fill" style="width: ${scorePct}%"></div>
            </div>
          </div>
        </div>
        <div class="chunk-excerpt">${excerptHtml}</div>
        <div class="chunk-footer">
          <span>Chunk ${chunk.index_in_doc + 1} · ${chunk.word_count} words</span>
          <button class="btn-copy-excerpt" data-chunk-text="${escapeHtml(chunk.content)}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>
              <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
            </svg>
            Copy
          </button>
        </div>
      `;

      const copyBtn = card.querySelector('.btn-copy-excerpt');
      copyBtn.addEventListener('click', () => {
        const textToCopy = chunk.content;
        navigator.clipboard.writeText(textToCopy).then(() => {
          showToast(`Copied chunk [${cid}] excerpt.`);
        });
      });

      evidenceChunksFeed.appendChild(card);
    });
  }

  function openEvidenceDrawer() {
    evidenceInspector.classList.remove('collapsed');
    btnToggleEvidence.classList.add('active');
  }

  function closeEvidenceDrawer() {
    evidenceInspector.classList.add('collapsed');
    btnToggleEvidence.classList.remove('active');
  }

  function focusCitationInEvidenceDrawer(citationId) {
    openEvidenceDrawer();

    setTimeout(() => {
      const targetCard = document.getElementById(`evidence-chunk-${citationId}`);
      if (targetCard) {
        document.querySelectorAll('.chunk-card').forEach(c => c.classList.remove('target-highlight'));
        targetCard.classList.add('target-highlight');
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 100);
  }

  btnToggleEvidence.addEventListener('click', () => {
    if (evidenceInspector.classList.contains('collapsed')) {
      openEvidenceDrawer();
    } else {
      closeEvidenceDrawer();
    }
  });

  btnCloseEvidence.addEventListener('click', () => {
    closeEvidenceDrawer();
  });

  btnToggleSidebar.addEventListener('click', () => {
    corpusShelf.classList.toggle('open');
  });

  // -------------------------------------------------------------
  // Settings Dialog & Hyperparameters
  // -------------------------------------------------------------
  btnOpenSettings.addEventListener('click', () => {
    settingProvider.value = state.settings.provider || 'builtin';
    settingModel.value = state.settings.model || 'auto';
    settingTopK.value = state.settings.top_k || 4;
    valTopKDisplay.textContent = settingTopK.value;
    settingThreshold.value = state.settings.similarity_threshold || 0.10;
    valThresholdDisplay.textContent = Number(settingThreshold.value).toFixed(2);
    settingTemp.value = state.settings.temperature || 0.2;
    valTempDisplay.textContent = settingTemp.value;
    settingsStatusMsg.textContent = '';

    syncProviderFields();
    settingsModal.classList.add('open');
  });

  btnCloseSettings.addEventListener('click', () => {
    settingsModal.classList.remove('open');
  });

  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
      settingsModal.classList.remove('open');
    }
  });

  settingProvider.addEventListener('change', syncProviderFields);

  function syncProviderFields() {
    const isBuiltin = settingProvider.value === 'builtin';
    const apiKeyGroup = document.getElementById('api-key-group');
    const modelGroup = document.getElementById('model-select-group');

    if (isBuiltin) {
      apiKeyGroup.style.opacity = '0.5';
      settingApiKey.disabled = true;
      settingApiKey.placeholder = 'Not required for KORTEX Core';
      modelGroup.style.opacity = '0.5';
      settingModel.disabled = true;
    } else {
      apiKeyGroup.style.opacity = '1';
      settingApiKey.disabled = false;
      settingApiKey.placeholder = `Enter ${settingProvider.value.toUpperCase()} API Key`;
      modelGroup.style.opacity = '1';
      settingModel.disabled = false;
    }
  }

  btnToggleKeyVisibility.addEventListener('click', () => {
    if (settingApiKey.type === 'password') {
      settingApiKey.type = 'text';
      btnToggleKeyVisibility.textContent = 'Hide';
    } else {
      settingApiKey.type = 'password';
      btnToggleKeyVisibility.textContent = 'Show';
    }
  });

  settingTopK.addEventListener('input', (e) => {
    valTopKDisplay.textContent = e.target.value;
  });

  settingThreshold.addEventListener('input', (e) => {
    valThresholdDisplay.textContent = Number(e.target.value).toFixed(2);
  });

  settingTemp.addEventListener('input', (e) => {
    valTempDisplay.textContent = e.target.value;
  });

  settingsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      provider: settingProvider.value,
      api_key: settingApiKey.value.trim(),
      model: settingModel.value.trim() || 'auto',
      top_k: parseInt(settingTopK.value, 10),
      similarity_threshold: parseFloat(settingThreshold.value),
      temperature: parseFloat(settingTemp.value)
    };

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const json = await res.json();
      if (json.success) {
        state.settings = { ...state.settings, ...json.settings };
        settingsStatusMsg.textContent = '✓ Configuration saved.';
        paramTopkDisplay.textContent = `${state.settings.top_k} chunks`;
        paramThresholdDisplay.textContent = Number(state.settings.similarity_threshold).toFixed(2);

        const engineLabel = {
          'builtin': 'KORTEX Core',
          'gemini': 'Gemini 1.5',
          'openai': 'GPT-4o Mini',
          'groq': 'Groq Llama'
        }[state.settings.provider] || 'KORTEX Core';
        statEngineName.textContent = engineLabel;

        setTimeout(() => {
          settingsModal.classList.remove('open');
        }, 600);
      } else {
        settingsStatusMsg.textContent = json.error || 'Failed to save settings.';
      }
    } catch (err) {
      console.error(err);
      settingsStatusMsg.textContent = 'Network error saving settings.';
    }
  });

  // -------------------------------------------------------------
  // Helpers: Toast Notification & HTML Escaping
  // -------------------------------------------------------------
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = 'toast';
    if (type === 'error') {
      toast.style.borderLeftColor = 'var(--rose-danger)';
    }
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  function escapeHtml(text) {
    if (!text) return '';
    const map = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
  }

  // -------------------------------------------------------------
  // Initial Boot
  // -------------------------------------------------------------
  fetchStatus();
});
