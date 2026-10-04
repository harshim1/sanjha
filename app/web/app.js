// Sanjha offline web app. Everything on the Card, Sell-or-wait and Share tabs
// is computed on this phone from the last model + card it downloaded; only
// the SMS simulator and the cooperative tab need the server.
"use strict";

const $ = (id) => document.getElementById(id);
const VILLAGES = ["Ondera", "Kiptoo", "Marwa", "Serem"];
const HORIZONS = [2, 4, 8];
const WEEKS_PER_MONTH = 4.345;
const DRAWS = 3000;
let model = null, card = null, preview = null;

// ---------- loading: network first, last good copy otherwise ----------
async function getJSON(url, key) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    try { localStorage.setItem(key, JSON.stringify(data)); } catch (e) {}
    return { data, live: !res.headers.get("X-Sanjha-Saved") };
  } catch (e) {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key)); } catch (e2) {}
    return { data: saved, live: false };
  }
}

async function load() {
  const village = $("village").value;
  const m = await getJSON("/api/model.json", "sanjha-model");
  const c = await getJSON("/api/card?village=" + encodeURIComponent(village), "sanjha-card-" + village);
  model = m.data; card = c.data;
  const live = m.live && c.live;
  $("net").textContent = live ? "online" : "offline — saved card";
  $("net").className = "pill" + (live ? "" : " off");
  if (!model || !card) {
    $("card-sw").textContent = "No saved card yet. Connect once to download it.";
    $("card-en").textContent = $("card-meta").textContent = "";
    return;
  }
  // while there is signal, save every village's card for later
  if (live) VILLAGES.filter((v) => v !== village).forEach((v) => getJSON("/api/card?village=" + encodeURIComponent(v), "sanjha-card-" + v));
  preview = { gap: card.gap, v: card.var, n: 0 };
  $("offer").value = Math.round(card.fair / 5) * 5;
  renderCard(); renderFan(); renderChoices(); renderShare();
}

// ---------- seeded RNG + bootstrap of future prices ----------
function rng(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// h-week log shocks: sum of h resampled, symmetrized standardized weekly shocks
function futurePrices(fair, sigmaWeek, h, seed) {
  const pool = model.z_pool, r = rng(seed), out = new Float64Array(DRAWS);
  for (let i = 0; i < DRAWS; i++) {
    let z = 0;
    for (let j = 0; j < h; j++) { const v = pool[(r() * pool.length) | 0]; z += r() < 0.5 ? v : -v; }
    out[i] = fair * Math.exp(sigmaWeek * z);
  }
  return out.sort();
}
const quantile = (sorted, q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];

// ---------- card ----------
function renderCard() {
  $("card-sw").textContent = card.text;
  $("card-en").textContent = card.english;
  const basis = card.basis === "region"
    ? `regional estimate (${card.reporters} of ${5} farmers needed have shared in ${card.village})`
    : `${card.reporters} farmers in ${card.village} have shared, pooled with the region`;
  $("card-meta").textContent = `Week of ${card.week_of} · ${basis} · ${card.text.length}/160 characters`;
  const synth = Object.entries(model.sources).filter(([, v]) => v.includes("synthetic")).map(([k]) => k.replace(/_/g, " "));
  $("sources").textContent = synth.length ? "Demo data: " + synth.join(", ") + " are synthetic. Futures: " + model.sources.futures.join(", ") + "." : "";
}

// ---------- voice: stitch pre-generated clips, all served from the offline cache ----------
let voice = null;
function numberClips(n) {           // mirrors number_clips() in voice/build_clips.py
  const parts = [["h", Math.floor(n / 100)], ["t", Math.floor((n % 100) / 10)], ["u", n % 10]].filter(([, d]) => d).map(([p, d]) => p + d);
  return parts.flatMap((p, i) => (i ? ["p_and", p] : [p]));
}
function cardClips() {
  if (card.abstain) return ["p_abstain"];
  return ["p_fair", ...numberClips(card.lo), "p_and", ...numberClips(card.hi), "p_perkg",
    "p_below", ...numberClips(card.floor), "p_ask", "p_right", ...numberClips(card.hits), "p_outof", ...numberClips(card.n)];
}
async function loadVoice() {
  const { data, live } = await getJSON("/voice/manifest.json", "sanjha-voice");
  voice = data && data.clips ? data : null;
  $("listen").hidden = !voice;
  if (voice && live) voice.clips.forEach((c) => fetch(`/voice/${c}.mp3`).catch(() => {}));  // warm the offline cache
}
async function speakCard() {
  if (!voice || !card) return;
  $("listen").disabled = true;
  for (const c of cardClips()) {
    await new Promise((done) => { const a = new Audio(`/voice/${c}.mp3`); a.onended = a.onerror = done; a.play().catch(done); });
  }
  $("listen").disabled = false;
}

// ---------- fan chart ----------
function renderFan() {
  const cv = $("fan"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  const css = getComputedStyle(document.documentElement);
  const fg = css.getPropertyValue("--fg"), muted = css.getPropertyValue("--muted"), line = css.getPropertyValue("--line"), accent = css.getPropertyValue("--accent");
  const hist = model.history.map((p) => p.ref * Math.exp(-card.gap));
  const bands = [];
  for (let h = 1; h <= 8; h++) {
    const d = futurePrices(card.fair, card.sigma_week, h, 100 + h);
    bands.push([0.1, 0.25, 0.75, 0.9].map((q) => quantile(d, q)));
  }
  const all = hist.concat(bands.flat(), [card.fair]);
  const lo = Math.min(...all) * 0.97, hi = Math.max(...all) * 1.03;
  const n = hist.length + 8, pad = { l: 44, r: 10, t: 10, b: 24 };
  const x = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H);
  ctx.font = "12px system-ui"; ctx.fillStyle = muted; ctx.strokeStyle = line; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const v = lo + (g / 4) * (hi - lo);
    ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(W - pad.r, y(v)); ctx.stroke();
    ctx.fillText(Math.round(v / 10) * 10, 6, y(v) + 4);
  }
  ctx.fillText("26 weeks ago", pad.l, H - 6); ctx.fillText("now", x(hist.length - 1) - 10, H - 6); ctx.fillText("+8 wk", W - 44, H - 6);
  const now = hist.length - 1;
  const band = (a, b, alpha) => {
    ctx.beginPath(); ctx.moveTo(x(now), y(card.fair));
    bands.forEach((q, i) => ctx.lineTo(x(now + 1 + i), y(q[b])));
    for (let i = bands.length - 1; i >= 0; i--) ctx.lineTo(x(now + 1 + i), y(bands[i][a]));
    ctx.closePath(); ctx.globalAlpha = alpha; ctx.fillStyle = accent; ctx.fill(); ctx.globalAlpha = 1;
  };
  band(0, 3, 0.18); band(1, 2, 0.3);
  ctx.strokeStyle = accent; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x(now), y(card.fair)); ctx.lineTo(x(n - 1), y(card.fair)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = fg; ctx.lineWidth = 1.8; ctx.beginPath();
  hist.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.lineTo(x(now), y(card.fair)); ctx.stroke();
}

// ---------- sell now / sell half / wait ----------
function renderChoices() {
  const offer = +$("offer").value, monthly = +$("rate").value / 100;
  $("offer-out").textContent = offer; $("rate-out").textContent = (monthly * 100).toFixed(1) + "%";
  const p = model.decision_params, r = Math.pow(1 + monthly, 1 / WEEKS_PER_MONTH) - 1;
  let rows = `<tr><th>Choice</th><th class="num">Typical</th><th class="num">Bad case</th><th class="num">Chance it beats selling now</th></tr>
    <tr><td>Sell all now</td><td class="num">${offer}</td><td class="num">${offer}</td><td class="num">—</td></tr>`;
  if (card.abstain) {
    rows += `<tr><td colspan="4">${card.text}</td></tr>`;
  } else {
    for (const h of HORIZONS) {
      const keep = Math.pow(1 - p.storage_loss_per_week, h), cost = p.cost_per_kg_per_week * h, disc = Math.pow(1 + r, h);
      const wait = futurePrices(card.fair, card.sigma_week, h, 7 + h).map((f) => (f * keep - cost) / disc);
      const beats = wait.filter((v) => v > offer).length / wait.length;
      const med = quantile(wait, 0.5), bad = quantile(wait, 0.1);
      rows += `<tr><td>Sell half, wait ${h} weeks</td><td class="num">${Math.round((offer + med) / 2)}</td><td class="num">${Math.round((offer + bad) / 2)}</td><td class="num">${Math.round(beats * 100)}%</td></tr>
        <tr><td>Wait ${h} weeks</td><td class="num">${Math.round(med)}</td><td class="num">${Math.round(bad)}</td><td class="num">${Math.round(beats * 100)}%</td></tr>`;
    }
  }
  $("choices").innerHTML = rows;
}

// ---------- Vand Chhako preview: the Kalman update, on the phone ----------
function rangeFor(gap, v) {
  const fair = model.futures_usd_lb * model.fx * model.k * Math.exp(-gap);
  const s = Math.sqrt(model.sigma_week_global ** 2 + model.kalman.q + v), q = model.conformal_q;
  return [fair * Math.exp(-q * s), fair * Math.exp(q * s)];
}
function kalmanUpdate(price) {
  const y = Math.log(model.futures_usd_lb * model.fx * model.k) - Math.log(price), r = model.kalman.r;
  const u = (y - preview.gap) / Math.sqrt(preview.v + r) / 3;                 // robust: surprising offers count less
  const w = Math.abs(u) < 1 ? (1 - u * u) ** 2 : 0;
  if (w > 1e-6) { const K = preview.v / (preview.v + r / w); preview.gap += K * (y - preview.gap); preview.v *= 1 - K; }
  preview.n += 1;
  return w;
}
function renderShare(lastWeight) {
  const before = rangeFor(card.gap, card.var), after = rangeFor(preview.gap, preview.v);
  const lo = Math.min(before[0], after[0]) * 0.95, hi = Math.max(before[1], after[1]) * 1.05;
  const bar = (label, r) => `<div class="bar"><div class="cap"><span>${label}</span><span>${Math.round(r[0])}–${Math.round(r[1])} /kg (width ${Math.round(r[1] - r[0])})</span></div>
    <div class="track"><div class="fill" style="left:${((r[0] - lo) / (hi - lo)) * 100}%;width:${((r[1] - r[0]) / (hi - lo)) * 100}%"></div></div></div>`;
  let note = "";
  if (lastWeight !== undefined) note = lastWeight === 0 ? "That offer is far from everything else shared, so it was set aside."
    : lastWeight < 0.5 ? "That offer is unusual, so it counted for less." : "";
  $("share-bars").innerHTML = bar("Before", before) + bar(`After ${preview.n} more shared offer${preview.n === 1 ? "" : "s"}`, after) + `<p class="meta">${note}</p>`;
}

// ---------- SMS simulator ----------
function bubble(cls, text, small) {
  const d = document.createElement("div"); d.className = "msg " + cls; d.textContent = text;
  if (small) { const s = document.createElement("small"); s.textContent = small; d.appendChild(s); }
  $("thread").appendChild(d); $("thread").scrollTop = 1e6;
}
async function say(text) {
  if (!text.trim()) return;
  bubble("out", text);
  try {
    const res = await fetch("/sms/incoming", { method: "POST", body: new URLSearchParams({ from: $("sim-from").value, text }) });
    const r = await res.json();
    bubble("in", r.reply, `${r.reply_english} · intent: ${r.intent} (${Math.round(r.confidence * 100)}%)`);
  } catch (e) { bubble("in", "No signal — the simulator needs the server. The saved card on the Card tab still works."); }
}

// ---------- cooperative dashboard ----------
async function renderCoop() {
  const { data: d, live } = await getJSON("/api/dashboard", "sanjha-dashboard");
  if (!d) { $("coop-meta").textContent = "Dashboard needs a connection."; return; }
  $("villages").innerHTML = `<tr><th>Village</th><th class="num">Farmers sharing</th><th class="num">Offers</th><th class="num">Set aside</th><th>Estimate from</th><th class="num">Range /kg</th></tr>` +
    d.villages.map((v) => `<tr><td>${v.village}</td><td class="num">${v.reporters}</td><td class="num">${v.reports}</td><td class="num">${v.ignored}</td><td>${v.basis === "region" ? `region (under ${d.k_anonymity})` : "village + region"}</td><td class="num">${v.lo}–${v.hi}</td></tr>`).join("");
  $("reports").innerHTML = `<tr><th>Village</th><th class="num">Offer</th><th class="num">Weight</th><th>Via</th></tr>` +
    d.recent_reports.map((r) => `<tr><td>${r.village}</td><td class="num">${Math.round(r.price)}</td><td class="num">${r.weight}</td><td>${r.source}</td></tr>`).join("");
  const c = d.classifier;
  $("coop-meta").textContent = `${live ? "" : "Offline — last saved view. "}Incoming messages by intent: ${JSON.stringify(d.incoming_by_intent)}. ` +
    `Intent classifier: ${Math.round(c.held_out_accuracy * 100)}% on ${c.held_out_n} hand-written test messages (trained on synthetic data). No phone numbers are stored.`;
}

// ---------- wiring ----------
VILLAGES.forEach((v) => $("village").add(new Option(v, v)));
$("village").onchange = load;
document.querySelectorAll("nav button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("nav button, .tab").forEach((el) => el.classList.remove("on"));
  b.classList.add("on"); $(b.dataset.tab).classList.add("on");
  if (b.dataset.tab === "coop") renderCoop();
}));
$("offer").oninput = $("rate").oninput = () => card && renderChoices();
$("share-add").onclick = () => { const p = +$("share-price").value; if (card && p >= 50 && p <= 3000) renderShare(kalmanUpdate(p)); };
$("share-reset").onclick = () => { if (!card) return; preview = { gap: card.gap, v: card.var, n: 0 }; renderShare(); };
$("sim-form").onsubmit = (e) => { e.preventDefault(); say($("sim-text").value); $("sim-text").value = ""; };
document.querySelectorAll("[data-say]").forEach((b) => (b.onclick = () => say(b.dataset.say)));
window.addEventListener("online", load); window.addEventListener("offline", load);
$("listen").onclick = speakCard;
loadVoice();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");
load();
