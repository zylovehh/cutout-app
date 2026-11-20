// 抠图 App：本地模型算整图透明度，点一下挑要留下的连通块，画笔微调（擦除/恢复），边缘羽化，存透明底 PNG。
// 照片只在手机里算，不上传。

// ===== 模型配置（部署时填好） =====
const MODEL = {
  url: 'rmbg14-fp32.onnx',
  bytes: 176153355,
  label: 'RMBG-1.4 全精度',
  sizeMB: '168',
  norm: 'rmbg',   // 'rmbg' | 'imagenet'
  out: 'minmax'   // 'minmax' | 'sigmoid'
};

const $ = (id) => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
const PROC_EDGE = 1024;      // 处理时长边
const SAVE_EDGE = 1600;      // 保存时长边上限
const UNDO_MAX = 6;

let origImg = null;          // HTMLImageElement
let origW = 0, origH = 0;
let procW = 0, procH = 0;    // 处理尺寸（长边 1024）
let session = null;          // ort session
let alphaDone = false;
let fullAlpha = null;        // Float32Array(procW*procH)，0..1
let compId = null;           // Int32Array，0=背景，其余为块编号
let compCount = 0;
let compSize = [];           // 每块像素数
let selSet = new Set();      // 选中的块编号
let undoStack = [];          // Array<{sel: Set, delta: Int8Array}>
let baseMask = null;         // Uint8Array(procW*procH)，0/255，点选结果（已扩 2px）
let editDelta = null;        // Int8Array，-100..100，画笔改动（擦除负，恢复正）
let effMask = null;          // Uint8Array，0..255，base + delta 后的有效掩膜
let featheredMask = null;    // Uint8Array，0..255，羽化后的掩膜（羽化为 0 时等于 effMask）
let mode = 'pick';           // 'pick' | 'erase' | 'restore'
let brushSize = 36;          // 画笔直径（处理尺寸像素）
let feather = 2;             // 羽化半径（处理尺寸像素）
let stroking = false;        // 画笔是否正在拖动

// ---------- 小工具 ----------
let toastTimer = null;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg; t.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.style.display = 'none'; }, 2600);
}
function setStep(n) {
  [1, 2, 3].forEach(i => $('step' + i).classList.toggle('on', i <= n));
}
function showProgress(show, pct, txt) {
  $('progress').style.display = show ? 'block' : 'none';
  if (pct != null) $('pfill').style.width = pct + '%';
  if (txt != null) $('ptxt').textContent = txt;
}

if (location.protocol === 'file:') $('openTip').style.display = 'block';
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
$('modelNote').textContent =
  '首次使用要下载一次模型（约 ' + MODEL.sizeMB + ' MB，' + MODEL.label + '），下载后存在手机里，下次打开不用再下。';

// ---------- 选图 ----------
const IMG_SUFFIX = ['jpg', 'jpeg', 'png', 'heic', 'heif', 'webp', 'gif', 'bmp', 'avif'];
$('fileInput').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) loadFile(f);
  e.target.value = '';
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) loadFile(f);
});

function loadFile(file) {
  const name = (file.name || '').toLowerCase();
  const suffix = name.split('.').pop();
  const okType = (file.type || '').startsWith('image/');
  const okSuffix = IMG_SUFFIX.includes(suffix);
  if (!okType && !okSuffix) { toast('这不是图片，换一张照片试试'); return; }
  const rd = new FileReader();
  rd.onload = () => {
    const im = new Image();
    im.onload = () => setupImage(im);
    im.onerror = () => toast('这张照片打不开，换一张试试');
    im.src = rd.result;
  };
  rd.onerror = () => toast('读照片失败，换一张试试');
  rd.readAsDataURL(file);
}

function setupImage(im) {
  origImg = im; origW = im.naturalWidth; origH = im.naturalHeight;
  const s = PROC_EDGE / Math.max(origW, origH);
  procW = Math.max(1, Math.round(origW * s));
  procH = Math.max(1, Math.round(origH * s));
  alphaDone = false; fullAlpha = null; compId = null; compCount = 0;
  selSet = new Set(); undoStack = [];
  baseMask = null; effMask = null; featheredMask = null;
  editDelta = new Int8Array(procW * procH);
  stroking = false;
  cv.width = procW; cv.height = procH;
  ctx.drawImage(origImg, 0, 0, procW, procH);
  $('editor').style.display = 'block';
  $('undoBtn').disabled = true;
  setStep(2);
  setMode('pick');
  $('hint').textContent = '正在准备…点一下要留下的地方，模型第一次要下载，稍等一下。';
  $('pickBtn').style.display = 'none';
  $('editor').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- 模型 ----------
async function ensureSession() {
  if (session) return session;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = './';
  showProgress(true, 0, '正在下载模型… 0%（约 ' + MODEL.sizeMB + ' MB）');
  const res = await fetch(MODEL.url);
  if (!res.ok) throw new Error('模型下载失败 ' + res.status);
  const total = parseInt(res.headers.get('Content-Length') || '', 10) || MODEL.bytes || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.byteLength;
    if (total > 0) {
      const pct = Math.min(99, Math.round(got / total * 100));
      showProgress(true, pct, '正在下载模型… ' + pct + '%（约 ' + MODEL.sizeMB + ' MB）');
    }
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
  showProgress(true, 100, '模型下好了，正在加载…');
  session = await ort.InferenceSession.create(buf, {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all'
  });
  showProgress(false);
  return session;
}

// ---------- 整图透明度 ----------
async function computeAlpha() {
  if (alphaDone) return;
  await ensureSession();
  showProgress(true, null, '正在抠图…第一次要算一会儿，请稍等');
  const c = document.createElement('canvas');
  c.width = 1024; c.height = 1024;
  const cx = c.getContext('2d');
  cx.drawImage(origImg, 0, 0, 1024, 1024);
  const d = cx.getImageData(0, 0, 1024, 1024).data;
  const input = new Float32Array(3 * 1024 * 1024);
  if (MODEL.norm === 'imagenet') {
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      input[j] = (d[i] / 255 - mean[0]) / std[0];
      input[1024 * 1024 + j] = (d[i + 1] / 255 - mean[1]) / std[1];
      input[2 * 1024 * 1024 + j] = (d[i + 2] / 255 - mean[2]) / std[2];
    }
  } else {
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      input[j] = d[i] / 255 - 0.5;
      input[1024 * 1024 + j] = d[i + 1] / 255 - 0.5;
      input[2 * 1024 * 1024 + j] = d[i + 2] / 255 - 0.5;
    }
  }
  const tensor = new ort.Tensor('float32', input, [1, 3, 1024, 1024]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  let m = out[session.outputNames[0]].data;
  m = Float32Array.from(m);
  if (MODEL.out === 'sigmoid') {
    for (let i = 0; i < m.length; i++) m[i] = 1 / (1 + Math.exp(-m[i]));
  } else {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < m.length; i++) { if (m[i] < mn) mn = m[i]; if (m[i] > mx) mx = m[i]; }
    const range = (mx - mn) || 1;
    for (let i = 0; i < m.length; i++) m[i] = (m[i] - mn) / range;
  }
  // 1024x1024 掩膜缩回处理尺寸
  const mc = document.createElement('canvas');
  mc.width = 1024; mc.height = 1024;
  const mctx = mc.getContext('2d');
  const mid = mctx.createImageData(1024, 1024);
  for (let i = 0, j = 0; i < m.length; i++, j += 4) {
    const v = Math.round(Math.max(0, Math.min(1, m[i])) * 255);
    mid.data[j] = v; mid.data[j + 1] = v; mid.data[j + 2] = v; mid.data[j + 3] = 255;
  }
  mctx.putImageData(mid, 0, 0);
  const pc = document.createElement('canvas');
  pc.width = procW; pc.height = procH;
  const pctx = pc.getContext('2d');
  pctx.drawImage(mc, 0, 0, procW, procH);
  const pd = pctx.getImageData(0, 0, procW, procH).data;
  fullAlpha = new Float32Array(procW * procH);
  for (let i = 0, j = 0; i < fullAlpha.length; i++, j += 4) fullAlpha[i] = pd[j] / 255;
  labelComponents();
  alphaDone = true;
  showProgress(false);
}

// ---------- 连通块 ----------
function labelComponents() {
  const N = procW * procH;
  compId = new Int32Array(N);
  const minSize = Math.max(40, N * 0.0004);
  const stack = new Int32Array(N);
  let next = 0;
  const sizes = [0];
  for (let s = 0; s < N; s++) {
    if (fullAlpha[s] < 0.5 || compId[s] !== 0) continue;
    next++;
    let sp = 0, sz = 0;
    stack[sp++] = s; compId[s] = next;
    while (sp > 0) {
      const p = stack[--sp]; sz++;
      const x = p % procW, y = (p / procW) | 0;
      if (x > 0 && fullAlpha[p - 1] >= 0.5 && compId[p - 1] === 0) { compId[p - 1] = next; stack[sp++] = p - 1; }
      if (x < procW - 1 && fullAlpha[p + 1] >= 0.5 && compId[p + 1] === 0) { compId[p + 1] = next; stack[sp++] = p + 1; }
      if (y > 0 && fullAlpha[p - procW] >= 0.5 && compId[p - procW] === 0) { compId[p - procW] = next; stack[sp++] = p - procW; }
      if (y < procH - 1 && fullAlpha[p + procW] >= 0.5 && compId[p + procW] === 0) { compId[p + procW] = next; stack[sp++] = p + procW; }
    }
    sizes[next] = sz;
  }
  // 丢小碎块，编号重排
  const remap = new Int32Array(next + 1);
  let kept = 0;
  for (let i = 1; i <= next; i++) if (sizes[i] >= minSize) remap[i] = ++kept;
  compSize = [0];
  for (let i = 1; i <= next; i++) if (remap[i]) compSize[remap[i]] = sizes[i];
  for (let p = 0; p < N; p++) if (compId[p]) compId[p] = remap[compId[p]];
  compCount = kept;
}

function compAt(x, y) {
  x = Math.max(0, Math.min(procW - 1, x | 0));
  y = Math.max(0, Math.min(procH - 1, y | 0));
  const hit = compId[y * procW + x];
  if (hit) return hit;
  const r = Math.max(2, Math.round(procW * 0.04));
  for (let d = 1; d <= r; d++) {
    for (let dy = -d; dy <= d; dy++) {
      for (let dx = -d; dx <= d; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== d) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= procW || ny < 0 || ny >= procH) continue;
        const v = compId[ny * procW + nx];
        if (v) return v;
      }
    }
  }
  return 0;
}

// ---------- 选区 + 画笔 + 羽化 ----------
function pushUndo() {
  undoStack.push({ sel: new Set(selSet), delta: editDelta.slice() });
  if (undoStack.length > UNDO_MAX) undoStack.shift();
  $('undoBtn').disabled = undoStack.length === 0;
}

function applySelection() {
  const N = procW * procH;
  let mask = new Uint8Array(N);
  for (let p = 0; p < N; p++) if (selSet.has(compId[p])) mask[p] = 255;
  // 掩膜扩 2px：横向再纵向各做两次 3 点最大值
  for (let pass = 0; pass < 2; pass++) {
    const tmp = new Uint8Array(N);
    for (let y = 0; y < procH; y++) {
      const row = y * procW;
      for (let x = 0; x < procW; x++) {
        let v = mask[row + x];
        if (x > 0 && mask[row + x - 1] > v) v = mask[row + x - 1];
        if (x < procW - 1 && mask[row + x + 1] > v) v = mask[row + x + 1];
        tmp[row + x] = v;
      }
    }
    mask = tmp;
    const tmp2 = new Uint8Array(N);
    for (let y = 0; y < procH; y++) {
      const row = y * procW;
      for (let x = 0; x < procW; x++) {
        let v = mask[row + x];
        if (y > 0 && mask[row + x - procW] > v) v = mask[row + x - procW];
        if (y < procH - 1 && mask[row + x + procW] > v) v = mask[row + x + procW];
        tmp2[row + x] = v;
      }
    }
    mask = tmp2;
  }
  baseMask = mask;
  recomputeEff();
  applyFeather();
}

// base + 画笔改动 = 有效掩膜（不含羽化）
function recomputeEff() {
  const N = procW * procH;
  effMask = new Uint8Array(N);
  for (let p = 0; p < N; p++) {
    let v = baseMask[p] + Math.round(editDelta[p] * 2.55);
    effMask[p] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

// 羽化：用画布自带的高斯模糊糊一下掩膜边缘
function blurMask(mask, w, h, r) {
  if (r <= 0) return mask;
  const c1 = document.createElement('canvas'); c1.width = w; c1.height = h;
  const x1 = c1.getContext('2d');
  const img = x1.createImageData(w, h);
  for (let p = 0, j = 0; p < mask.length; p++, j += 4) {
    img.data[j] = 255; img.data[j + 1] = 255; img.data[j + 2] = 255; img.data[j + 3] = mask[p];
  }
  x1.putImageData(img, 0, 0);
  const c2 = document.createElement('canvas'); c2.width = w; c2.height = h;
  const x2 = c2.getContext('2d');
  if (!('filter' in x2)) return mask;
  x2.filter = 'blur(' + r + 'px)';
  x2.drawImage(c1, 0, 0);
  x2.filter = 'none';
  const d = x2.getImageData(0, 0, w, h).data;
  const out = new Uint8Array(w * h);
  for (let p = 0, j = 0; p < out.length; p++, j += 4) out[p] = d[j + 3];
  return out;
}

function applyFeather() {
  featheredMask = blurMask(effMask, procW, procH, feather);
}

function currentMask() {
  return featheredMask || effMask;
}

// 画笔涂一笔：软边圆笔刷，中间实、边缘渐弱
function brushDab(cx, cy) {
  const r = brushSize / 2;
  const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(procW - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(procH - 1, Math.ceil(cy + r));
  const dir = mode === 'erase' ? -1 : 1;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      const d = Math.sqrt(dx * dx + dy * dy) / r;
      if (d >= 1) continue;
      const f = 1 - d;
      const p = y * procW + x;
      let v = editDelta[p] + dir * f * 100;
      editDelta[p] = v < -100 ? -100 : v > 100 ? 100 : Math.round(v);
    }
  }
  recomputeEff();
}

// 把原图 + 掩膜合成 RGBA，边缘半透明像素用周围不透明像素的颜色去白晕
function buildRGBA(srcCanvas, mask, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const cx = c.getContext('2d');
  cx.drawImage(srcCanvas, 0, 0, w, h);
  const img = cx.getImageData(0, 0, w, h);
  const d = img.data;
  const N = w * h;
  for (let p = 0; p < N; p++) d[p * 4 + 3] = mask[p];
  // 去白晕：半透明像素取周围不透明像素的平均颜色
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x, a = d[p * 4 + 3];
      if (a === 0 || a === 255) continue;
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const q = ny * w + nx;
          if (d[q * 4 + 3] === 255) { r += d[q * 4]; g += d[q * 4 + 1]; b += d[q * 4 + 2]; n++; }
        }
      }
      if (n > 0) { d[p * 4] = r / n; d[p * 4 + 1] = g / n; d[p * 4 + 2] = b / n; }
    }
  }
  return img;
}

function render() {
  const m = currentMask();
  if (!m) { ctx.drawImage(origImg, 0, 0, procW, procH); return; }
  const src = document.createElement('canvas');
  src.width = procW; src.height = procH;
  src.getContext('2d').drawImage(origImg, 0, 0, procW, procH);
  ctx.putImageData(buildRGBA(src, m, procW, procH), 0, 0);
}

// ---------- 模式 ----------
function setMode(m) {
  mode = m;
  $('modePick').classList.toggle('on', m === 'pick');
  $('modeErase').classList.toggle('on', m === 'erase');
  $('modeRestore').classList.toggle('on', m === 'restore');
  // 画笔要拖着涂，拖动时不让页面跟着滚；点选模式保持原来的手势
  cv.style.touchAction = m === 'pick' ? 'manipulation' : 'none';
  if (m === 'pick') {
    $('hint').textContent = selSet.size
      ? '已选 ' + selSet.size + ' 处。点一下去掉一处，或点“保存透明 PNG”。'
      : '点一下要留下的地方；再点一下，去掉它。';
  } else if (m === 'erase') {
    $('hint').textContent = '涂掉不要的部分（比如海报上的大字）。涂错了切到“恢复”涂回来。';
  } else {
    $('hint').textContent = '涂回被误擦掉的地方，也可以直接涂出模型没抠到的部分。';
  }
}
$('modePick').addEventListener('click', () => setMode('pick'));
$('modeErase').addEventListener('click', () => setMode('erase'));
$('modeRestore').addEventListener('click', () => setMode('restore'));

$('brushSize').addEventListener('input', (e) => {
  brushSize = +e.target.value;
  $('brushSizeVal').textContent = brushSize;
});
$('feather').addEventListener('input', (e) => {
  feather = +e.target.value;
  $('featherVal').textContent = feather;
  if (effMask) { applyFeather(); render(); }
});

// ---------- 画布手势 ----------
function canvasPos(e) {
  const rect = cv.getBoundingClientRect();
  return [(e.clientX - rect.left) / rect.width * procW, (e.clientY - rect.top) / rect.height * procH];
}

let busy = false;
cv.addEventListener('pointerdown', async (e) => {
  if (!origImg || busy) return;
  e.preventDefault();
  busy = true;
  try {
    await computeAlpha();
    const [x, y] = canvasPos(e);
    if (mode !== 'pick') {
      // 画笔模式：按住拖着涂
      if (!baseMask) applySelection();
      pushUndo();
      stroking = true;
      try { cv.setPointerCapture(e.pointerId); } catch (_) {}
      brushDab(x, y);
      featheredMask = effMask; // 拖动时先不羽化，抬手再算，免得卡
      render();
      return;
    }
    // 点选模式：点一下开关一块
    const id = compAt(x, y);
    if (!id) { toast('这里没有抠出来的东西，点一下要留下的地方'); return; }
    pushUndo();
    if (selSet.has(id)) selSet.delete(id); else selSet.add(id);
    applySelection();
    render();
    setStep(3);
    if (selSet.size) {
      $('hint').textContent = '已选 ' + selSet.size + ' 处。点一下去掉一处，或点“保存透明 PNG”。';
    } else {
      $('hint').textContent = '都去掉了，点“撤销”找回来，或者点“自动抠图”重来。';
      toast('都去掉了，点“撤销”找回来');
    }
  } catch (err) {
    console.error(err);
    toast('抠图失败了，检查一下网络再试一次');
    showProgress(false);
  } finally {
    busy = false;
  }
});

cv.addEventListener('pointermove', (e) => {
  if (!stroking || !origImg) return;
  e.preventDefault();
  const [x, y] = canvasPos(e);
  brushDab(x, y);
  featheredMask = effMask;
  render();
});

function endStroke() {
  if (!stroking) return;
  stroking = false;
  applyFeather();
  render();
}
cv.addEventListener('pointerup', endStroke);
cv.addEventListener('pointercancel', endStroke);

$('autoBtn').addEventListener('click', async () => {
  if (!origImg || busy) return;
  busy = true;
  try {
    await computeAlpha();
    pushUndo();
    selSet = new Set();
    for (let i = 1; i <= compCount; i++) selSet.add(i);
    editDelta.fill(0);
    applySelection();
    render();
    setStep(3);
    $('hint').textContent = '已选全部 ' + compCount + ' 处。点一下去掉不要的那处，或用“擦除”涂掉不要的部分。';
  } catch (err) {
    console.error(err);
    toast('抠图失败了，检查一下网络再试一次');
    showProgress(false);
  } finally {
    busy = false;
  }
});

$('undoBtn').addEventListener('click', () => {
  const prev = undoStack.pop();
  if (!prev) return;
  selSet = prev.sel;
  editDelta = prev.delta;
  $('undoBtn').disabled = undoStack.length === 0;
  applySelection();
  render();
  setStep(selSet.size ? 3 : 2);
  $('hint').textContent = selSet.size
    ? '已选 ' + selSet.size + ' 处。点一下去掉一处，或点“保存透明 PNG”。'
    : '点一下要留下的地方；再点一下，去掉它。';
});

$('resetBtn').addEventListener('click', () => {
  if (!selSet.size && !effMask) return;
  pushUndo();
  selSet = new Set();
  editDelta.fill(0);
  applySelection();
  render();
  $('hint').textContent = '点一下要留下的地方；再点一下，去掉它。';
});

$('saveBtn').addEventListener('click', () => {
  if (!origImg) return;
  const m = currentMask();
  let any = false;
  if (m) for (let p = 0; p < m.length; p++) { if (m[p] > 0) { any = true; break; } }
  if (!any) { toast('现在什么都没留下，先点一下要留下的地方'); return; }
  const s = Math.min(1, SAVE_EDGE / Math.max(origW, origH));
  const sw = Math.max(1, Math.round(origW * s));
  const sh = Math.max(1, Math.round(origH * s));
  // 掩膜缩放到保存尺寸
  const mc = document.createElement('canvas');
  mc.width = procW; mc.height = procH;
  const mid = mc.getContext('2d').createImageData(procW, procH);
  for (let p = 0, j = 0; p < m.length; p++, j += 4) {
    mid.data[j] = 255; mid.data[j + 1] = 255; mid.data[j + 2] = 255; mid.data[j + 3] = m[p];
  }
  mc.getContext('2d').putImageData(mid, 0, 0);
  const sc = document.createElement('canvas');
  sc.width = sw; sc.height = sh;
  sc.getContext('2d').drawImage(mc, 0, 0, sw, sh);
  const sd = sc.getContext('2d').getImageData(0, 0, sw, sh).data;
  const saveMask = new Uint8Array(sw * sh);
  for (let p = 0, j = 0; p < saveMask.length; p++, j += 4) saveMask[p] = sd[j + 3];
  const src = document.createElement('canvas');
  src.width = origW; src.height = origH;
  src.getContext('2d').drawImage(origImg, 0, 0);
  const out = document.createElement('canvas');
  out.width = sw; out.height = sh;
  out.getContext('2d').putImageData(buildRGBA(src, saveMask, sw, sh), 0, 0);
  out.toBlob((blob) => {
    if (!blob) { toast('保存失败，再试一次'); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = '抠图-透明底.png';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    toast('已保存，去文件里找');
  }, 'image/png');
});

$('againBtn').addEventListener('click', () => {
  $('editor').style.display = 'none';
  $('pickBtn').style.display = 'block';
  $('fileInput').click();
});
