"use strict";

// ----------------------------------------------------------------------------
// Small client for the origami fold simulator. Talks to the stdlib HTTP backend
// in origami/ui/server.py and renders the board + paper on a canvas.
// All board geometry is in metres; the UI shows centimetres for convenience.
// ----------------------------------------------------------------------------

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const errorEl = document.getElementById("error");
const hintEl = document.getElementById("hint");

let state = null;          // latest workspace state from the server
let pending = [];          // board-coordinate points for the fold-in-progress
let view = null;           // current board->pixel transform

const M2CM = 100;
// Tolerance for treating a clicked point as lying exactly on the fold line.
const ON_LINE_TOLERANCE = 1e-9;

// ---- server calls ----------------------------------------------------------
async function api(path, body) {
  const opts = { method: body ? "POST" : "GET" };
  if (body) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `request failed (${res.status})`);
  }
  return data;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

async function call(path, body) {
  try {
    showError("");
    state = await api(path, body);
    render();
  } catch (err) {
    showError(err.message);
  }
}

// ---- coordinate transforms -------------------------------------------------
function computeView() {
  const margin = 36;
  const bw = state.board.width;
  const bh = state.board.height;
  const scale = Math.min(
    (canvas.width - 2 * margin) / bw,
    (canvas.height - 2 * margin) / bh
  );
  const offX = (canvas.width - bw * scale) / 2;
  const offY = (canvas.height - bh * scale) / 2;
  view = { scale, offX, offY, bh };
}

function boardToPx(x, y) {
  return [view.offX + x * view.scale, canvas.height - view.offY - y * view.scale];
}

function pxToBoard(px, py) {
  return [
    (px - view.offX) / view.scale,
    (canvas.height - view.offY - py) / view.scale,
  ];
}

// Convert a DOM mouse event to internal canvas pixel coordinates.
function eventToCanvas(evt) {
  const rect = canvas.getBoundingClientRect();
  return [
    (evt.clientX - rect.left) * (canvas.width / rect.width),
    (evt.clientY - rect.top) * (canvas.height / rect.height),
  ];
}

// ---- rendering -------------------------------------------------------------
function render() {
  computeView();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBoard();
  drawPaper();
  drawBackside();
  drawCreases();
  drawMagnets();
  drawLandmarks();
  drawPending();
  renderLandmarkList();
  renderHistory();
  updateControls();
}

function drawBoard() {
  const [x0, y0] = boardToPx(0, 0);
  const [x1, y1] = boardToPx(state.board.width, state.board.height);
  ctx.fillStyle = "#101b27";
  ctx.fillRect(x1 < x0 ? x1 : x0, y1 < y0 ? y1 : y0, Math.abs(x1 - x0), Math.abs(y1 - y0));
  ctx.strokeStyle = "#33475f";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);

  ctx.fillStyle = "#5d7088";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("board origin (0, 0)", x0 + 4, y0 - 6);
}

function drawPaper() {
  const poly = state.paper.polygon;
  if (poly.length < 2) return;
  ctx.beginPath();
  poly.forEach((p, i) => {
    const [px, py] = boardToPx(p[0], p[1]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(244, 241, 232, 0.92)";
  ctx.fill();
  ctx.strokeStyle = "#b9b29a";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawBackside() {
  // Folded-over flaps expose the paper's reverse face, drawn yellow instead of
  // being left transparent. Later folds sit on top of earlier ones.
  state.paper.folds.forEach((fold) => {
    (fold.flaps || []).forEach((flap) => {
      if (!flap || flap.length < 3) return;
      ctx.beginPath();
      flap.forEach((p, i) => {
        const [px, py] = boardToPx(p[0], p[1]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = "#f2d23a";
      ctx.fill();
      ctx.strokeStyle = "#b9b29a";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  });
}

function drawCreases() {
  state.paper.folds.forEach((fold) => {
    const [sx, sy] = boardToPx(fold.start[0], fold.start[1]);
    const [ex, ey] = boardToPx(fold.end[0], fold.end[1]);
    ctx.beginPath();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = fold.style === "mountain" ? "#e0683a" : "#2f7de0";
    ctx.lineWidth = 1.6;
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

function drawMagnets() {
  (state.magnets || []).forEach((m) => {
    const [mx, my] = boardToPx(m.x, m.y);
    ctx.fillStyle = "#8a6bd1";
    ctx.fillRect(mx - 5, my - 5, 10, 10);
  });
}

function drawLandmarks() {
  ctx.font = "11px system-ui, sans-serif";
  state.paper.landmarks.forEach((lm) => {
    const [px, py] = boardToPx(lm.x, lm.y);
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "#1d2a3a";
    ctx.fill();
    ctx.strokeStyle = "#4f9cff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = "#cdd9e8";
    ctx.fillText(lm.name, px + 6, py - 5);
  });
}

function drawPending() {
  if (pending.length === 0) return;
  ctx.fillStyle = "#ffce6b";
  pending.forEach((p) => {
    const [px, py] = boardToPx(p[0], p[1]);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  });
  if (pending.length === 2) {
    // Draw the (extended) fold line so the user can see both sides.
    const a = pending[0];
    const b = pending[1];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const span = (state.board.width + state.board.height);
    const [sx, sy] = boardToPx(a[0] - ux * span, a[1] - uy * span);
    const [ex, ey] = boardToPx(b[0] + ux * span, b[1] + uy * span);
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#ffce6b";
    ctx.lineWidth = 1.4;
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function renderLandmarkList() {
  const ul = document.getElementById("landmarks");
  ul.innerHTML = "";
  state.paper.landmarks.forEach((lm) => {
    const li = document.createElement("li");
    li.innerHTML = `<b>${lm.name}</b><span>${(lm.x * M2CM).toFixed(1)}, ${(lm.y * M2CM).toFixed(1)} cm</span>`;
    ul.appendChild(li);
  });
}

function renderHistory() {
  const ol = document.getElementById("history");
  ol.innerHTML = "";
  state.history.forEach((entry) => {
    const li = document.createElement("li");
    li.textContent = entry;
    ol.appendChild(li);
  });
}

function updateControls() {
  document.getElementById("undoBtn").disabled = !state.can_undo;
  document.getElementById("redoBtn").disabled = !state.can_redo;
  const cancelBtn = document.getElementById("cancelFold");
  cancelBtn.hidden = pending.length === 0;
  if (pending.length === 0) {
    hintEl.textContent = "Click two points on the board to set a fold line, then click the side you want to fold over.";
  } else if (pending.length === 1) {
    hintEl.textContent = "Click the second point of the fold line.";
  } else {
    hintEl.textContent = "Click on the side of the line you want to fold across the crease.";
  }
}

// ---- fold interaction ------------------------------------------------------
// Signed side of point relative to the directed line a->b (matches the sign
// convention of FoldLine.side_of on the server).
function sideOf(a, b, p) {
  const cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  if (Math.abs(cross) < ON_LINE_TOLERANCE) return 0;
  return cross > 0 ? 1 : -1;
}

canvas.addEventListener("click", (evt) => {
  if (!state) return;
  const [cx, cy] = eventToCanvas(evt);
  const point = pxToBoard(cx, cy);
  if (pending.length < 2) {
    pending.push(point);
    render();
    return;
  }
  // Third click: pick the moving side and apply the fold.
  const side = sideOf(pending[0], pending[1], point);
  if (side === 0) {
    showError("Pick a point clearly to one side of the fold line.");
    return;
  }
  const body = {
    p1: pending[0],
    p2: pending[1],
    moving_side: side,
    target_point: point,
    style: document.getElementById("foldStyle").value,
    label: document.getElementById("foldLabel").value,
  };
  pending = [];
  document.getElementById("foldLabel").value = "";
  call("/api/fold", body);
});

// ---- control wiring --------------------------------------------------------
document.getElementById("shape").addEventListener("change", (e) => {
  document.getElementById("heightField").style.display =
    e.target.value === "square" ? "none" : "flex";
});

document.getElementById("newBtn").addEventListener("click", () => {
  const shape = document.getElementById("shape").value;
  const width = parseFloat(document.getElementById("newWidth").value) / M2CM;
  const heightVal = parseFloat(document.getElementById("newHeight").value) / M2CM;
  const height = shape === "square" ? width : heightVal;
  pending = [];
  call("/api/new", { shape, width, height });
});

document.getElementById("rotateBtn").addEventListener("click", () => {
  const angle = parseFloat(document.getElementById("rotateDeg").value);
  call("/api/rotate", { angle_deg: angle });
});

document.getElementById("translateBtn").addEventListener("click", () => {
  const dx = parseFloat(document.getElementById("transDx").value) / M2CM;
  const dy = parseFloat(document.getElementById("transDy").value) / M2CM;
  call("/api/translate", { dx, dy });
});

document.getElementById("dartBtn").addEventListener("click", () => {
  pending = [];
  call("/api/dart", {});
});

document.getElementById("undoBtn").addEventListener("click", () => {
  pending = [];
  call("/api/undo", {});
});

document.getElementById("redoBtn").addEventListener("click", () => {
  pending = [];
  call("/api/redo", {});
});

document.getElementById("resetBtn").addEventListener("click", () => {
  pending = [];
  call("/api/reset", {});
});

document.getElementById("cancelFold").addEventListener("click", () => {
  pending = [];
  showError("");
  render();
});

// ---- boot ------------------------------------------------------------------
call("/api/state");
