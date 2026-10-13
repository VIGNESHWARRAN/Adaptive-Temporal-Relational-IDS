/**
 * ADAPTIVE IDS NEURAL SOC - CLIENT APPLICATION LOGIC
 * Real-time File-Based Network Flow Triggering & Threat Monitoring
 */

// Application State
const state = {
  batchSize: 1,
  activeFile: '',
  availableFiles: [],
  currentOffset: 0,
  totalRows: 0,
  isStreaming: false,
  streamIntervalId: null,
  streamSpeedMs: 1000,
  filterMode: 'all', // 'all' | 'attacks' | 'good'
  searchQuery: '',
  records: [],
  chartHistory: [], // max 40 data points for live stream canvas
};

// DOM References
const el = {
  btnTrigger: document.getElementById('btnTrigger'),
  btnAutoStream: document.getElementById('btnAutoStream'),
  streamIcon: document.getElementById('streamIcon'),
  streamLabel: document.getElementById('streamLabel'),
  streamInterval: document.getElementById('streamInterval'),
  btnNCount: document.getElementById('btnNCount'),
  cliBatchDisplay: document.getElementById('cliBatchDisplay'),
  inputBatchN: document.getElementById('inputBatchN'),
  btnDecN: document.getElementById('btnDecN'),
  btnIncN: document.getElementById('btnIncN'),
  presetPills: document.querySelectorAll('.preset-pill'),
  selectCsvFile: document.getElementById('selectCsvFile'),
  btnResetPointer: document.getElementById('btnResetPointer'),
  fileUploadInput: document.getElementById('fileUploadInput'),
  txtOffset: document.getElementById('txtOffset'),
  txtTotalRows: document.getElementById('txtTotalRows'),
  pointerProgressFill: document.getElementById('pointerProgressFill'),
  
  // Threat banner
  threatAlertBar: document.getElementById('threatAlertBar'),
  threatIcon: document.getElementById('threatIcon'),
  threatHeadline: document.getElementById('threatHeadline'),
  threatSubtext: document.getElementById('threatSubtext'),
  threatScoreDisplay: document.getElementById('threatScoreDisplay'),
  defconLevelDisplay: document.getElementById('defconLevelDisplay'),

  // KPIs
  kpiTotalFlows: document.getElementById('kpiTotalFlows'),
  kpiGoodFlows: document.getElementById('kpiGoodFlows'),
  kpiGoodPct: document.getElementById('kpiGoodPct'),
  kpiAttackFlows: document.getElementById('kpiAttackFlows'),
  kpiAttackPct: document.getElementById('kpiAttackPct'),
  kpiLatency: document.getElementById('kpiLatency'),
  activeModelBadge: document.getElementById('activeModelBadge'),

  // Visualizations
  tempModPct: document.getElementById('tempModPct'),
  tempModBar: document.getElementById('tempModBar'),
  relModPct: document.getElementById('relModPct'),
  relModBar: document.getElementById('relModBar'),
  attackTagsList: document.getElementById('attackTagsList'),
  flowStreamCanvas: document.getElementById('flowStreamCanvas'),

  // Table
  flowsTableBody: document.getElementById('flowsTableBody'),
  feedCountBadge: document.getElementById('feedCountBadge'),
  filterTabs: document.querySelectorAll('.filter-tab'),
  feedSearchInput: document.getElementById('feedSearchInput'),
  btnClearFeed: document.getElementById('btnClearFeed'),

  // Modal
  modalBackdrop: document.getElementById('modalBackdrop'),
  btnModalClose: document.getElementById('btnModalClose'),
  modalFlowTitle: document.getElementById('modalFlowTitle'),
  modalFlowSubtitle: document.getElementById('modalFlowSubtitle'),
  modalVerdictBanner: document.getElementById('modalVerdictBanner'),
  modalVerdictPrimary: document.getElementById('modalVerdictPrimary'),
  modalVerdictSecondary: document.getElementById('modalVerdictSecondary'),
  modalGoodProb: document.getElementById('modalGoodProb'),
  modalAttackProb: document.getElementById('modalAttackProb'),
  modalTempWeight: document.getElementById('modalTempWeight'),
  modalRelWeight: document.getElementById('modalRelWeight'),
  modalTelemetryGrid: document.getElementById('modalTelemetryGrid'),

  // Toast
  toastAlert: document.getElementById('toastAlert'),
  toastIcon: document.getElementById('toastIcon'),
  toastMsg: document.getElementById('toastMsg'),
};

// ==============================================================================
// INITIALIZATION
// ==============================================================================
async function initApp() {
  setupEventListeners();
  initCanvas();
  await fetchStatus();
  renderChart();
}

// Fetch Initial State from Backend
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    
    state.batchSize = data.batch_size || 1;
    state.activeFile = data.active_file || '';
    state.availableFiles = data.available_files || [];
    state.currentOffset = data.pointer_offset || 0;
    state.totalRows = data.total_rows || 0;

    // Update UI elements
    updateBatchDisplay(state.batchSize);
    updatePointerUI(state.currentOffset, state.totalRows);
    populateFileDropdown(state.availableFiles, state.activeFile);

    if (data.dataset_name) {
      el.activeModelBadge.textContent = `${data.dataset_name} (${data.fusion_type || 'concat'})`;
    }

    if (data.stats) {
      updateKpiCards(data.stats);
      updateThreatAlert(data.stats);
    }
  } catch (err) {
    console.error('Failed to fetch status:', err);
    showToast('⚠️ Could not connect to IDS engine backend.', 'error');
  }
}

// Populate CSV Dropdown
function populateFileDropdown(files, active) {
  el.selectCsvFile.innerHTML = '';
  if (!files || files.length === 0) {
    el.selectCsvFile.innerHTML = '<option disabled>No CSV files found in incoming_flows/</option>';
    return;
  }

  files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f;
    if (f === active) opt.selected = true;
    el.selectCsvFile.appendChild(opt);
  });
}

// Update Pointer Display
function updatePointerUI(offset, total) {
  el.txtOffset.textContent = offset.toLocaleString();
  el.txtTotalRows.textContent = total.toLocaleString();
  const pct = total > 0 ? Math.min(100, Math.round((offset / total) * 100)) : 0;
  el.pointerProgressFill.style.width = `${pct}%`;
}

// Update Batch Size UI
function updateBatchDisplay(n) {
  el.inputBatchN.value = n;
  el.btnNCount.textContent = n;
  el.cliBatchDisplay.textContent = `n = ${n}`;
  
  el.presetPills.forEach(p => {
    p.classList.toggle('active', parseInt(p.dataset.n) === n);
  });
}

// ==============================================================================
// CORE TRIGGER LOGIC
// ==============================================================================
async function triggerFlows(nOverride = null) {
  const n = nOverride !== null ? nOverride : state.batchSize;
  
  try {
    const res = await fetch('/api/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n: n, filename: state.activeFile })
    });

    const data = await res.json();
    if (!data.success) {
      showToast(data.error || 'Trigger failed', 'error');
      return;
    }

    // Process new batch records
    const newRecords = data.records || [];
    if (newRecords.length > 0) {
      // Add to front of state records
      state.records = [...newRecords, ...state.records].slice(0, 500);
      
      // Update pointer
      if (data.pointer) {
        state.currentOffset = data.pointer.current_offset;
        state.totalRows = data.pointer.total_rows;
        updatePointerUI(state.currentOffset, state.totalRows);
      }

      // Update KPIs & Threat Alert
      if (data.stats) {
        updateKpiCards(data.stats);
        updateThreatAlert(data.stats);
      }

      // Check for attacks in this batch to show toast notification
      const attacksInBatch = newRecords.filter(r => r.is_attack);
      if (attacksInBatch.length > 0) {
        const firstAtk = attacksInBatch[0];
        showToast(`🚨 ATTACK DETECTED: ${firstAtk.attack_signature} (Conf: ${firstAtk.confidence}%)`, 'attack');
      }

      // Update Chart points
      newRecords.forEach(r => {
        state.chartHistory.push({
          isAttack: r.is_attack,
          prob: r.attack_prob,
          time: r.timestamp
        });
      });
      if (state.chartHistory.length > 40) {
        state.chartHistory = state.chartHistory.slice(-40);
      }

      // Update modality bars from latest record
      const latest = newRecords[0];
      if (latest) {
        el.tempModPct.textContent = `${latest.temporal_modality_pct}%`;
        el.tempModBar.style.width = `${latest.temporal_modality_pct}%`;
        el.relModPct.textContent = `${latest.relational_modality_pct}%`;
        el.relModBar.style.width = `${latest.relational_modality_pct}%`;
      }

      renderTable();
      renderChart();
    }
  } catch (err) {
    console.error('Trigger request error:', err);
    showToast('Failed to trigger flow batch.', 'error');
  }
}

// ==============================================================================
// KPI & THREAT LEVEL RENDERING
// ==============================================================================
function updateKpiCards(stats) {
  const total = stats.total_inspected || 0;
  const good = stats.good_count || 0;
  const attacks = stats.attack_count || 0;

  el.kpiTotalFlows.textContent = total.toLocaleString();
  el.kpiGoodFlows.textContent = good.toLocaleString();
  el.kpiAttackFlows.textContent = attacks.toLocaleString();

  const goodPct = total > 0 ? ((good / total) * 100).toFixed(1) : '0.0';
  const atkPct = total > 0 ? ((attacks / total) * 100).toFixed(1) : '0.0';

  el.kpiGoodPct.textContent = `${goodPct}%`;
  el.kpiAttackPct.textContent = `${atkPct}%`;

  el.kpiLatency.innerHTML = `${stats.avg_latency_ms || 0.0} <span class="unit">ms</span>`;
  el.feedCountBadge.textContent = `${state.records.length} flows in feed`;

  // Update attack distribution tags
  const dist = stats.attack_distribution || {};
  const tagsKeys = Object.keys(dist);
  if (tagsKeys.length === 0) {
    el.attackTagsList.innerHTML = '<span class="empty-tag">No attacks triggered yet. System All Clear.</span>';
  } else {
    el.attackTagsList.innerHTML = tagsKeys.map(k => `
      <div class="attack-tag">
        <span>${escapeHtml(k)}</span>
        <span class="count">${dist[k]}</span>
      </div>
    `).join('');
  }
}

function updateThreatAlert(stats) {
  const level = stats.threat_level || 'NORMAL';
  const score = stats.threat_score || 0.0;

  el.threatScoreDisplay.textContent = `${score}%`;

  // Remove prior defcon classes
  el.threatAlertBar.className = 'threat-alert-bar';

  if (level === 'NORMAL') {
    el.threatAlertBar.classList.add('defcon-normal');
    el.threatIcon.textContent = '🟢';
    el.threatHeadline.textContent = 'DEFCON 5: NORMAL TRAFFIC - ALL SECURE';
    el.threatSubtext.textContent = 'Zero active malicious intrusions in recent flow buffer.';
    el.defconLevelDisplay.textContent = 'DEFCON 5';
    el.defconLevelDisplay.style.color = 'var(--emerald-safe)';
  } else if (level === 'GUARDED') {
    el.threatAlertBar.classList.add('defcon-guarded');
    el.threatIcon.textContent = '🔵';
    el.threatHeadline.textContent = 'DEFCON 4: GUARDED - MINOR ANOMALIES DETECTED';
    el.threatSubtext.textContent = 'Occasional attack signatures detected. Firewall actively logging.';
    el.defconLevelDisplay.textContent = 'DEFCON 4';
    el.defconLevelDisplay.style.color = 'var(--blue-neon)';
  } else if (level === 'ELEVATED') {
    el.threatAlertBar.classList.add('defcon-elevated');
    el.threatIcon.textContent = '🟡';
    el.threatHeadline.textContent = 'DEFCON 2: ELEVATED THREAT LEVEL';
    el.threatSubtext.textContent = 'High frequency of intrusion attempts detected in network stream.';
    el.defconLevelDisplay.textContent = 'DEFCON 2';
    el.defconLevelDisplay.style.color = 'var(--amber-warn)';
  } else {
    el.threatAlertBar.classList.add('defcon-critical');
    el.threatIcon.textContent = '🚨';
    el.threatHeadline.textContent = 'DEFCON 1: CRITICAL INTRUSION ATTACK IN PROGRESS';
    el.threatSubtext.textContent = 'Active malicious flood/exploit detected by Temporal-Relational classifier!';
    el.defconLevelDisplay.textContent = 'DEFCON 1';
    el.defconLevelDisplay.style.color = 'var(--crimson-attack)';
  }
}

// ==============================================================================
// TABLE RENDERING & FILTERING
// ==============================================================================
function renderTable() {
  const filtered = state.records.filter(r => {
    // Mode filter
    if (state.filterMode === 'attacks' && !r.is_attack) return false;
    if (state.filterMode === 'good' && r.is_attack) return false;

    // Search query filter
    if (state.searchQuery) {
      const q = state.searchQuery.toLowerCase();
      const match = (
        r.id.toLowerCase().includes(q) ||
        r.src_ip.toLowerCase().includes(q) ||
        r.dst_ip.toLowerCase().includes(q) ||
        r.src_port.toLowerCase().includes(q) ||
        r.dst_port.toLowerCase().includes(q) ||
        r.attack_signature.toLowerCase().includes(q) ||
        r.protocol.toLowerCase().includes(q)
      );
      if (!match) return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    el.flowsTableBody.innerHTML = `
      <tr class="empty-state-row">
        <td colspan="10">
          <div class="empty-feed-box">
            <div class="empty-icon">${state.records.length > 0 ? '🔍' : '📡'}</div>
            <div class="empty-msg">${state.records.length > 0 ? 'No flows match the active filter' : 'Ready to trigger flows'}</div>
            <div class="empty-hint">Click <b>TRIGGER NEXT FLOWS</b> to ingest from CSV</div>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  el.flowsTableBody.innerHTML = filtered.map(r => {
    const verdictClass = r.is_attack ? 'verdict-attack' : 'verdict-good';
    const verdictIcon = r.is_attack ? '🚨' : '🛡️';
    const sigClass = r.is_attack ? 'is-attack' : '';
    const barColor = r.is_attack ? 'var(--crimson-attack)' : 'var(--emerald-safe)';

    return `
      <tr data-flow-id="${r.id}">
        <td class="time-cell">${r.timestamp}</td>
        <td class="flow-id-cell">${r.id}</td>
        <td class="ip-cell">${escapeHtml(r.src_ip)}:${escapeHtml(r.src_port)} <span class="cyan-text">&rarr;</span> ${escapeHtml(r.dst_ip)}:${escapeHtml(r.dst_port)}</td>
        <td><span class="key-hint">${escapeHtml(r.protocol)}</span></td>
        <td>${r.duration_ms} ms</td>
        <td>
          <span class="verdict-badge ${verdictClass}">
            ${verdictIcon} ${r.verdict}
          </span>
        </td>
        <td class="attack-signature-text ${sigClass}">${escapeHtml(r.attack_signature)}</td>
        <td class="conf-cell">
          <div class="conf-bar-wrap">
            <span>${r.confidence}%</span>
            <div class="conf-mini-track">
              <div class="conf-mini-fill" style="width: ${r.confidence}%; background: ${barColor};"></div>
            </div>
          </div>
        </td>
        <td class="lat-cell">${r.latency_ms} ms</td>
        <td>
          <button class="btn-inspect" onclick="openFlowModal('${r.id}')">Inspect</button>
        </td>
      </tr>
    `;
  }).join('');
}

// ==============================================================================
// MODAL FLOW FORENSIC INSPECTION
// ==============================================================================
window.openFlowModal = function(flowId) {
  const r = state.records.find(item => item.id === flowId);
  if (!r) return;

  el.modalFlowTitle.textContent = `Flow Forensic Analysis: ${r.id}`;
  el.modalFlowSubtitle.textContent = `${r.src_ip}:${r.src_port} -> ${r.dst_ip}:${r.dst_port} | Proto: ${r.protocol}`;

  el.modalVerdictBanner.className = 'modal-verdict-banner ' + (r.is_attack ? 'attack' : 'good');
  el.modalVerdictPrimary.textContent = r.is_attack ? `🚨 ATTACK DETECTED: ${r.attack_signature}` : '🛡️ BENIGN / NORMAL NETWORK FLOW';
  el.modalVerdictSecondary.textContent = `Prediction Confidence: ${r.confidence}% | Severity: ${r.severity} | Latency: ${r.latency_ms} ms`;

  el.modalGoodProb.textContent = `${r.good_prob}%`;
  el.modalAttackProb.textContent = `${r.attack_prob}%`;
  el.modalTempWeight.textContent = `${r.temporal_modality_pct}%`;
  el.modalRelWeight.textContent = `${r.relational_modality_pct}%`;

  // Raw features matrix
  const raw = r.raw_features || {};
  const keys = Object.keys(raw);
  el.modalTelemetryGrid.innerHTML = keys.map(k => `
    <div class="tele-item">
      <span class="tele-key">${escapeHtml(k)}</span>
      <span class="tele-val">${escapeHtml(raw[k])}</span>
    </div>
  `).join('');

  el.modalBackdrop.classList.add('open');
};

function closeModal() {
  el.modalBackdrop.classList.remove('open');
}

// ==============================================================================
// CANVAS REAL-TIME STREAM VISUALIZATION
// ==============================================================================
let canvasCtx = null;

function initCanvas() {
  const canvas = el.flowStreamCanvas;
  if (!canvas) return;
  canvasCtx = canvas.getContext('2d');
  
  // Set real resolution based on DPI
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = (rect.width || 600) * dpr;
  canvas.height = (rect.height || 180) * dpr;
  canvasCtx.scale(dpr, dpr);
}

function renderChart() {
  if (!canvasCtx) return;
  const canvas = el.flowStreamCanvas;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 600;
  const height = rect.height || 180;

  canvasCtx.clearRect(0, 0, width, height);

  // Background grid
  canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  canvasCtx.lineWidth = 1;

  for (let y = 30; y < height; y += 35) {
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, y);
    canvasCtx.lineTo(width, y);
    canvasCtx.stroke();
  }

  const history = state.chartHistory;
  if (history.length === 0) {
    canvasCtx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    canvasCtx.font = '12px JetBrains Mono';
    canvasCtx.textAlign = 'center';
    canvasCtx.fillText('Awaiting flow stream data...', width / 2, height / 2);
    return;
  }

  const stepX = width / Math.max(1, history.length - 1 || 1);

  // Draw smooth threat line
  canvasCtx.beginPath();
  history.forEach((pt, idx) => {
    const x = idx * stepX;
    // Map prob (0 to 100) to height
    const y = height - 20 - (pt.prob / 100) * (height - 40);
    if (idx === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
  });
  canvasCtx.strokeStyle = 'rgba(0, 242, 254, 0.4)';
  canvasCtx.lineWidth = 2;
  canvasCtx.stroke();

  // Draw glow points
  history.forEach((pt, idx) => {
    const x = idx * stepX;
    const y = height - 20 - (pt.prob / 100) * (height - 40);

    canvasCtx.beginPath();
    canvasCtx.arc(x, y, pt.isAttack ? 5 : 3.5, 0, Math.PI * 2);
    canvasCtx.fillStyle = pt.isAttack ? '#ff0055' : '#00f5a0';
    canvasCtx.shadowColor = pt.isAttack ? '#ff0055' : '#00f5a0';
    canvasCtx.shadowBlur = pt.isAttack ? 10 : 6;
    canvasCtx.fill();
    canvasCtx.shadowBlur = 0; // reset
  });
}

// ==============================================================================
// TOAST NOTIFICATION
// ==============================================================================
let toastTimeout = null;
function showToast(msg, type = 'info') {
  el.toastMsg.textContent = msg;
  el.toastIcon.textContent = type === 'attack' ? '🚨' : (type === 'error' ? '⚠️' : 'ℹ️');
  el.toastAlert.style.borderColor = type === 'attack' ? 'var(--crimson-attack)' : 'var(--cyan-neon)';

  el.toastAlert.classList.add('show');
  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    el.toastAlert.classList.remove('show');
  }, 3500);
}

// Helper
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ==============================================================================
// EVENT LISTENERS
// ==============================================================================
function setupEventListeners() {
  // Trigger button
  el.btnTrigger.addEventListener('click', () => triggerFlows());

  // Global Keyboard Shortcut: Spacebar triggers next batch
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
      e.preventDefault();
      triggerFlows();
    }
  });

  // Batch size input
  el.inputBatchN.addEventListener('change', async (e) => {
    const n = Math.max(1, parseInt(e.target.value) || 1);
    await updateBatchSizeServer(n);
  });

  // Steppers
  el.btnDecN.addEventListener('click', async () => {
    const n = Math.max(1, state.batchSize - 1);
    await updateBatchSizeServer(n);
  });

  el.btnIncN.addEventListener('click', async () => {
    const n = state.batchSize + 1;
    await updateBatchSizeServer(n);
  });

  // Preset pills
  el.presetPills.forEach(pill => {
    pill.addEventListener('click', async () => {
      const n = parseInt(pill.dataset.n);
      await updateBatchSizeServer(n);
    });
  });

  // Auto-stream Toggle
  el.btnAutoStream.addEventListener('click', toggleAutoStream);

  el.streamInterval.addEventListener('change', (e) => {
    state.streamSpeedMs = parseInt(e.target.value) || 1000;
    if (state.isStreaming) {
      clearInterval(state.streamIntervalId);
      state.streamIntervalId = setInterval(() => triggerFlows(), state.streamSpeedMs);
    }
  });

  // Ingestion File Dropdown
  el.selectCsvFile.addEventListener('change', async (e) => {
    const filename = e.target.value;
    try {
      const res = await fetch('/api/select_file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });
      const data = await res.json();
      state.activeFile = filename;
      state.currentOffset = data.pointer_offset || 0;
      state.totalRows = data.total_rows || 0;
      updatePointerUI(state.currentOffset, state.totalRows);
      showToast(`Switched active flow file to: ${filename}`, 'info');
    } catch (err) {
      console.error(err);
      showToast('Failed to select file', 'error');
    }
  });

  // Reset pointer
  el.btnResetPointer.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/reset_pointer', { method: 'POST' });
      const data = await res.json();
      state.currentOffset = 0;
      updatePointerUI(0, state.totalRows);
      showToast('Pointer reset to beginning of file (Row 0)', 'info');
    } catch (err) {
      console.error(err);
    }
  });

  // Clear feed
  el.btnClearFeed.addEventListener('click', async () => {
    try {
      await fetch('/api/clear_history', { method: 'POST' });
      state.records = [];
      state.chartHistory = [];
      renderTable();
      renderChart();
      await fetchStatus();
      showToast('Telemetry feed and session stats cleared', 'info');
    } catch (err) {
      console.error(err);
    }
  });

  // File Upload
  el.fileUploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      showToast(`Uploading ${file.name}...`, 'info');
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Uploaded and loaded: ${data.filename}`, 'info');
        await fetchStatus();
      } else {
        showToast('Upload failed', 'error');
      }
    } catch (err) {
      console.error(err);
      showToast('Error uploading CSV', 'error');
    }
  });

  // Filter tabs
  el.filterTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      el.filterTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.filterMode = tab.dataset.filter;
      renderTable();
    });
  });

  // Search input
  el.feedSearchInput.addEventListener('input', (e) => {
    state.searchQuery = e.target.value;
    renderTable();
  });

  // Modal close handlers
  el.btnModalClose.addEventListener('click', closeModal);
  el.modalBackdrop.addEventListener('click', (e) => {
    if (e.target === el.modalBackdrop) closeModal();
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });

  // Window resize for canvas
  window.addEventListener('resize', () => {
    initCanvas();
    renderChart();
  });
}

// Server batch size update
async function updateBatchSizeServer(n) {
  state.batchSize = n;
  updateBatchDisplay(n);
  try {
    await fetch('/api/batch_size', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n })
    });
  } catch (err) {
    console.error(err);
  }
}

// Auto Stream Toggle
function toggleAutoStream() {
  state.isStreaming = !state.isStreaming;

  if (state.isStreaming) {
    el.btnAutoStream.classList.add('streaming');
    el.streamIcon.textContent = '⏹';
    el.streamLabel.textContent = 'STOP STREAM';
    state.streamSpeedMs = parseInt(el.streamInterval.value) || 1000;
    state.streamIntervalId = setInterval(() => triggerFlows(), state.streamSpeedMs);
    showToast(`▶ Auto-streaming active: triggering ${state.batchSize} flows every ${state.streamSpeedMs / 1000}s`, 'info');
  } else {
    el.btnAutoStream.classList.remove('streaming');
    el.streamIcon.textContent = '▶';
    el.streamLabel.textContent = 'AUTO-STREAM';
    if (state.streamIntervalId) {
      clearInterval(state.streamIntervalId);
      state.streamIntervalId = null;
    }
    showToast('⏹ Auto-stream stopped', 'info');
  }
}

// Bootstrap on DOM load
document.addEventListener('DOMContentLoaded', initApp);
