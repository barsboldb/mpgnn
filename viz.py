"""Result visualizer — turns results/*.json into one self-contained HTML dashboard.

Scans every RunLogger JSON, embeds the data inline (no server, no CDN, works
offline) and renders three views over the same filtered slice:

  Curves  — per-epoch metric / loss / epoch-time lines for up to 8 selected runs,
            with crosshair tooltip, EMA smoothing and best-epoch markers.
  Compare — best-metric vs depth scatter (colored by layer type) and a ranked
            bar list of the filtered runs. The thesis-central depth view.
  Table   — sortable summary table of every filtered run (the no-hover twin of
            the charts; every plotted value is reachable here).

Usage:
    python viz.py                       # -> figures/report.html, then open it
    python viz.py --out dash.html       # custom output path
    python viz.py --open                # also open in the default browser
"""
from __future__ import annotations

import argparse
import json
import os
import webbrowser


def load_runs(results_dir: str = "results") -> list[dict]:
    """Read every RunLogger JSON into a compact record the dashboard can plot.

    Keeps EVERY per-epoch series in history (loss, test, train, ...) so any of
    them is plottable, and carries the full config / summary / timing for the
    per-run details modal. Files without a run_id (standalone-script schemas,
    e.g. ye_connectivity) are skipped.
    """
    runs = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            run = json.load(f)
        if "run_id" not in run:
            continue
        cfg = run.get("config", {})
        hist = run.get("history", [])
        summary = run.get("summary", {})
        timing = run.get("timing", {})
        layers = cfg.get("layers", [])

        keys: set[str] = set()
        for h in hist:
            keys |= set(h.keys()) - {"epoch"}
        series = {m: [h.get(m) for h in hist] for m in keys}

        # runs logged before the explicit `model` field: infer like GNNConfig does
        model = cfg.get("model")
        if not model:
            attn = cfg.get("tokenization") == "node_edge" \
                or any(l.get("type") == "global_attn" for l in layers)
            model = "transformer" if attn else "gnn"

        runs.append({
            "id": run.get("run_id", fname),
            "dataset": run.get("dataset", "?"),
            "params": run.get("params"),
            "task": cfg.get("task", "graph"),
            "model": model,
            "tokenization": cfg.get("tokenization", "node"),
            "features": cfg.get("node_features", "-"),
            "depth": len(layers),
            "layer_type": layers[0]["type"] if layers else "?",
            "hidden": cfg.get("hidden_channels", "-"),
            "lr": cfg.get("lr", "-"),
            "epochs": cfg.get("epochs", len(hist)),
            "best": summary.get("test", summary.get("val", None)),
            "best_epoch": summary.get("best_epoch", None),
            "ms_epoch": round(timing["mean_epoch_train_s"] * 1e3, 1)
                        if "mean_epoch_train_s" in timing else None,
            "infer_ms": (timing.get("inference") or {}).get(
                "per_graph_ms", (timing.get("inference") or {}).get("per_call_ms")),
            "ood": (run.get("ood") or {}).get("em"),
            # collapse diagnostics from the best-epoch eval (connectivity runs)
            "p_within": (run.get("best_eval") or {}).get("p_within"),
            "p_cross": (run.get("best_eval") or {}).get("p_cross"),
            "by_diam": run.get("diameter_breakdown"),     # {diam: {n, em, verdict}} | None
            "ep": [h["epoch"] for h in hist],
            "series": series,                 # {key: [values...]} for ALL logged series
            "meta": {k: v for k, v in {
                "config": cfg, "summary": summary,
                "best_eval": run.get("best_eval"), "ood": run.get("ood"),
                "diameter_breakdown": run.get("diameter_breakdown"),
                "provenance": run.get("provenance"), "timing": timing,
            }.items() if v},
        })
    return runs


def build_html(runs: list[dict]) -> str:
    data_json = json.dumps(runs, separators=(",", ":"))
    return _TEMPLATE.replace("/*DATA*/", data_json)


# ── The dashboard: one HTML file, embedded canvas charts, no dependencies ──────
# Colors are the dataviz reference dark-mode palette (validated: lightness band,
# chroma floor, >=3:1 on the #1a1a19 surface; CVD floor band covered by the
# named sidebar legend + table view). Chrome tokens from the same reference.
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mpGNN experiments</title>
<style>
  :root {
    --page:#0d0d0d; --surface:#1a1a19; --grid:#2c2c2a; --axis:#383835;
    --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --ring:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#199e70; --s3:#c98500; --s4:#008300;
    --s5:#9085e9; --s6:#e66767; --s7:#d55181; --s8:#d95926;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
         background:var(--page); color:var(--ink); }
  header { padding:12px 20px; border-bottom:1px solid var(--grid); display:flex;
           align-items:center; gap:16px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:650; }
  header .sub { color:var(--muted); font-size:12px; }
  .tabs { display:flex; gap:2px; margin-left:auto; background:var(--surface);
          border:1px solid var(--grid); border-radius:8px; padding:2px; }
  .tabs button { background:none; border:none; color:var(--ink2); padding:5px 14px;
                 border-radius:6px; font-size:12px; cursor:pointer; }
  .tabs button.on { background:var(--grid); color:var(--ink); font-weight:600; }
  .filters { display:flex; gap:8px; padding:10px 20px; border-bottom:1px solid var(--grid);
             flex-wrap:wrap; align-items:center; }
  select, input[type=search], button.ctl {
    background:var(--surface); color:var(--ink2); border:1px solid var(--grid);
    border-radius:6px; padding:5px 9px; font-size:12px; }
  select:hover, button.ctl:hover { border-color:var(--muted); cursor:pointer; }
  input[type=search] { width:180px; }
  .filters .spacer { flex:1; }
  .filters .note { color:var(--muted); font-size:11px; }
  .wrap { display:flex; height:calc(100vh - 92px); }
  .side { width:340px; min-width:280px; border-right:1px solid var(--grid);
          overflow-y:auto; padding:10px; }
  .side .bar { display:flex; gap:6px; margin-bottom:8px; }
  .main { flex:1; padding:16px 20px; overflow-y:auto; display:flex; flex-direction:column; }
  .run { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:7px;
         cursor:pointer; border:1px solid transparent; }
  .run:hover { background:var(--surface); }
  .run.on { background:var(--surface); border-color:var(--grid); }
  .sw { width:11px; height:11px; border-radius:3px; flex:none; background:var(--axis); }
  .run .meta { flex:1; min-width:0; }
  .run .name { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:12px;
               color:var(--ink); }
  .run .tags { color:var(--muted); font-size:11px; }
  .run .best { font-variant-numeric:tabular-nums; font-weight:650; font-size:12px;
               color:var(--ink2); }
  .run .info { background:none; border:none; color:var(--muted); font-size:14px;
               padding:0 2px; cursor:pointer; flex:none; }
  .run .info:hover { color:var(--ink); }
  .charts { display:grid; grid-template-columns:1fr; gap:16px; max-width:1000px; }
  .charts.grid2 { grid-template-columns:repeat(2, minmax(0,1fr)); max-width:none; }
  /* .fill grids stretch to the main area and split into two equal rows */
  .charts.fill { flex:1; min-height:0; }
  @media (min-width:1081px){
    .charts.fill { grid-template-rows:repeat(2, minmax(0,1fr)); }
    .card canvas.grow { flex:1; min-height:0; }
  }
  @media (max-width:1080px){
    .charts.grid2 { grid-template-columns:1fr; }
    .card canvas.grow { height:300px; }
  }
  .card { background:var(--surface); border:1px solid var(--ring); border-radius:10px;
          padding:12px 14px; display:flex; flex-direction:column; min-height:0; }
  .card h2 { margin:0 0 2px; font-size:13px; font-weight:600; display:flex;
             align-items:center; gap:8px; }
  .card .hint { color:var(--muted); font-size:11px; margin-bottom:6px; }
  canvas { width:100%; display:block; }
  .empty { color:var(--muted); padding:40px; text-align:center; }
  .tip { position:fixed; pointer-events:none; background:var(--page);
         border:1px solid var(--grid); border-radius:6px; padding:6px 9px; font-size:11px;
         opacity:0; transition:opacity .08s; z-index:10; max-width:340px; }
  .tip .v { color:var(--ink); font-weight:650; font-variant-numeric:tabular-nums; }
  .tip .k { display:inline-block; width:14px; height:0; border-top:2px solid;
            vertical-align:middle; margin-right:5px; border-radius:1px; }
  .tip .row { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--ink2); }
  table { border-collapse:collapse; width:100%; font-size:12px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--grid);
           white-space:nowrap; }
  td:first-child { max-width:330px; overflow:hidden; text-overflow:ellipsis; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  th { color:var(--muted); font-weight:600; cursor:pointer; user-select:none;
       position:sticky; top:0; background:var(--page); }
  th:hover { color:var(--ink); }
  tbody tr { cursor:pointer; }
  tbody tr:hover { background:var(--surface); }
  tbody tr.on td:first-child { box-shadow:inset 3px 0 0 var(--rowc, var(--axis)); }
  .legend { display:flex; gap:14px; flex-wrap:wrap; margin:4px 0 8px; font-size:11px;
            color:var(--ink2); }
  .legend .it { display:flex; align-items:center; gap:5px; }
  .legend .dot { width:9px; height:9px; border-radius:50%; }
  .modal-bg { position:fixed; inset:0; background:rgba(0,0,0,.62); z-index:50;
              display:none; align-items:center; justify-content:center; }
  .modal-bg.show { display:flex; }
  .modal { background:var(--surface); border:1px solid var(--grid); border-radius:10px;
           width:90%; max-width:760px; max-height:86vh; overflow:auto; padding:16px 18px; }
  .modal h3 { margin:0 0 6px; font-size:13px; word-break:break-all; }
  .modal .x { float:right; cursor:pointer; color:var(--muted); font-size:20px; line-height:1; }
  .modal .x:hover { color:var(--ink); }
  .modal .sec { color:var(--ink2); font-size:11px; font-weight:650; margin:0 0 6px;
                text-transform:uppercase; letter-spacing:.4px; }
  .modal .msec { margin:14px 0; }
  .modal .kv { display:grid; grid-template-columns:150px 1fr; row-gap:3px; column-gap:12px;
               font-size:12px; }
  .modal .kv .k { color:var(--muted); }
  .modal .kv .v { color:var(--ink2); font-variant-numeric:tabular-nums; word-break:break-word; }
  .modal .stats { display:flex; gap:22px; flex-wrap:wrap; }
  .modal .stat .sv { font-size:19px; font-weight:650; color:var(--ink); }
  .modal .stat .sk { font-size:11px; color:var(--muted); margin-top:1px; }
  .modal table.mini { width:auto; }
  .modal table.mini th, .modal table.mini td { padding:3px 14px 3px 0; font-size:12px;
                border-bottom:1px solid var(--grid); position:static; background:none; cursor:default; }
  .modal table.mini td { color:var(--ink2); }
  .modal .embar { height:3px; background:var(--s1); border-radius:2px; margin-top:2px; }
  .modal details { margin-top:14px; }
  .modal summary { color:var(--muted); font-size:12px; cursor:pointer; }
  .modal summary:hover { color:var(--ink); }
  .modal pre { background:var(--page); border:1px solid var(--grid); border-radius:6px;
               padding:10px; font-size:11px; overflow-x:auto; white-space:pre-wrap;
               word-break:break-word; margin:8px 0 0; }
  .toast { position:fixed; bottom:18px; left:50%; transform:translateX(-50%);
           background:var(--surface); border:1px solid var(--grid); color:var(--ink2);
           padding:8px 14px; border-radius:8px; font-size:12px; opacity:0;
           transition:opacity .2s; z-index:60; }
  .toast.show { opacity:1; }
  label.sl { color:var(--muted); font-size:11px; display:flex; align-items:center; gap:6px; }
</style>
</head>
<body>
<header>
  <h1>mpGNN experiments</h1>
  <span class="sub"><span id="cnt"></span> · <span id="hint"></span></span>
  <div class="tabs" id="tabs">
    <button data-v="overview" class="on">Overview</button>
    <button data-v="curves">Curves</button>
    <button data-v="compare">Compare</button>
    <button data-v="table">Table</button>
  </div>
</header>

<div class="filters">
  <select id="fDs"><option value="">all datasets</option></select>
  <select id="fTask"><option value="">all tasks</option></select>
  <select id="fLayer"><option value="">all layer types</option></select>
  <select id="fModel"><option value="">all architectures</option></select>
  <input type="search" id="fQ" placeholder="search runs…">
  <span class="spacer"></span>
  <span class="note">filters scope the run list and every view</span>
</div>

<div class="wrap">
  <aside class="side">
    <div class="bar">
      <button class="ctl" id="top8">select top 8</button>
      <button class="ctl" id="clear">clear</button>
    </div>
    <div id="list"></div>
  </aside>
  <main class="main" id="main"></main>
</div>

<div class="tip" id="tip"></div>
<div class="toast" id="toast"></div>
<div class="modal-bg" id="modalBg"><div class="modal" id="modal"></div></div>

<script>
const RUNS = /*DATA*/;

// dataviz reference palette, dark mode (validated against #1a1a19)
const SLOTS = ["#3987e5","#199e70","#c98500","#008300","#9085e9","#e66767","#d55181","#d95926"];
// fixed hue order per layer type — color follows the entity, never its rank
const LAYER_ORDER = ["global_attn","gat","gin","gcn","sage","bdh"];
const layerColor = t => SLOTS[Math.max(0, LAYER_ORDER.indexOf(t)) % 8];

const CHROME = { grid:"#2c2c2a", axis:"#383835", muted:"#898781", ink:"#ffffff", ink2:"#c3c2b7" };
// architecture family = the reasoning mechanism, not the host engine class.
// (Connectivity runs all report engine 'transformer'/'bdh' because the pairwise
// readout lives there; the layer type is what actually distinguishes them, so a
// GAT connectivity run is a message-passing GNN, not a transformer.)
function archOf(r){
  if(r.model==='bdh' || r.layer_type==='bdh') return 'BDH';
  if(r.layer_type==='global_attn') return 'transformer';
  if(['gat','gin','gcn','sage'].includes(r.layer_type)) return 'GNN';
  return r.model || '?';
}
RUNS.forEach(r => { r.slot = null; r.arch = archOf(r); });

const $ = id => document.getElementById(id);
const tip = $('tip');
let view = "overview", metric = "test", smooth = 0;
let sortKey = "best", sortDir = -1;

const fmt = v => v==null ? "–" : (Math.round(v*1000)/1000).toFixed(3);
const short = r => r.id.replace(/^\d{8}_\d{6}_/, '');

// ── filters ──────────────────────────────────────────────────────────────────
function fill(sel, vals){ [...new Set(vals)].sort().forEach(v => {
  const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }); }
fill($('fDs'),    RUNS.map(r=>r.dataset));
fill($('fTask'),  RUNS.map(r=>r.task));
fill($('fLayer'), RUNS.map(r=>r.layer_type));
fill($('fModel'), RUNS.map(r=>r.arch));

function filtered(){
  const ds=$('fDs').value, task=$('fTask').value, lay=$('fLayer').value,
        mod=$('fModel').value, q=$('fQ').value.toLowerCase();
  return RUNS.filter(r =>
    (!ds   || r.dataset===ds) && (!task || r.task===task) &&
    (!lay  || r.layer_type===lay) && (!mod || r.arch===mod) &&
    (!q    || r.id.toLowerCase().includes(q)));
}

// ── selection: up to 8 active runs, each keeps its slot while active ─────────
function activate(r){
  if (r.slot != null) { r.slot = null; return; }
  const used = new Set(RUNS.map(x=>x.slot).filter(s=>s!=null));
  for (let s=0; s<8; s++) if (!used.has(s)) { r.slot = s; return; }
  toast("Up to 8 runs can be compared at once — deselect one first.");
}
function active(){ return RUNS.filter(r=>r.slot!=null).sort((a,b)=>a.slot-b.slot); }
function toast(msg){ const t=$('toast'); t.textContent=msg; t.classList.add('show');
  clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove('show'), 2600); }

// ── metrics available in the current selection/filter ────────────────────────
const HIDE = new Set(["loss","train_time_s","eval_time_s"]);
const PRIORITY = ["test","verdict","train","p_within","p_cross","fp_cross","grad_norm","val"];
function metricList(){
  const s=new Set();
  RUNS.forEach(r=>Object.keys(r.series).forEach(k=>{ if(!HIDE.has(k)) s.add(k); }));
  return [...s].sort((a,b)=>{ const ia=PRIORITY.indexOf(a), ib=PRIORITY.indexOf(b);
    return (ia<0?99:ia)-(ib<0?99:ib) || a.localeCompare(b); });
}

// ── sidebar ──────────────────────────────────────────────────────────────────
function renderList(){
  const list=$('list'); list.textContent='';
  const fs = filtered();
  $('cnt').textContent = `${fs.length}/${RUNS.length} runs`;
  fs.forEach(r=>{
    const el=document.createElement('div');
    el.className='run'+(r.slot!=null?' on':'');
    const sw=document.createElement('span'); sw.className='sw';
    if (r.slot!=null) sw.style.background=SLOTS[r.slot];
    const meta=document.createElement('div'); meta.className='meta';
    const nm=document.createElement('div'); nm.className='name';
    nm.textContent=short(r); nm.title=r.id;
    const tg=document.createElement('div'); tg.className='tags';
    tg.textContent=`${r.dataset} · ${r.arch} · ${r.depth}×${r.layer_type} · h${r.hidden} · lr ${r.lr}`;
    meta.append(nm,tg);
    const best=document.createElement('div'); best.className='best'; best.textContent=fmt(r.best);
    const info=document.createElement('button'); info.className='info'; info.textContent='ⓘ';
    info.title='full config / summary / timing';
    info.onclick=e=>{ e.stopPropagation(); openModal(r); };
    el.append(sw,meta,best,info);
    el.onclick=()=>{ activate(r); renderAll(); };
    list.appendChild(el);
  });
}

// ── details modal ─────────────────────────────────────────────────────────────
const modalBg=$('modalBg'), modal=$('modal');

function el(tag, cls, text){
  const e=document.createElement(tag);
  if(cls) e.className=cls;
  if(text!=null) e.textContent=text;
  return e;
}
function fmtVal(v){
  if(v==null) return '–';
  if(typeof v==='boolean') return v?'yes':'no';
  if(typeof v==='number') return Number.isInteger(v)?v.toLocaleString():String(v);
  if(Array.isArray(v)||typeof v==='object') return JSON.stringify(v);
  return String(v);
}
function kvGrid(entries){
  // two-column definition grid; skips null/undefined values
  const g=el('div','kv');
  entries.forEach(([k,v])=>{
    if(v==null) return;
    g.append(el('span','k',k), el('span','v',fmtVal(v)));
  });
  return g;
}
function section(title, node){
  if(!node || !node.childElementCount) return null;   // nothing to show -> no section
  const wrap=el('div','msec');
  wrap.append(el('div','sec',title), node);
  return wrap;
}
function layersLine(layers){
  if(!layers||!layers.length) return null;
  const desc=l=>{
    const extras=Object.entries(l).filter(([k])=>k!=='type')
      .map(([k,v])=>`${k}=${v}`).join(', ');
    return l.type + (extras?` (${extras})`:'');
  };
  const first=JSON.stringify(layers[0]);
  return layers.every(l=>JSON.stringify(l)===first)
    ? `${layers.length} × ${desc(layers[0])}`
    : layers.map(desc).join('  →  ');
}
function statRow(stats){
  // small labeled figures in a row: the value leads, the label follows
  const row=el('div','stats');
  Object.entries(stats).forEach(([k,v])=>{
    if(v==null) return;
    const s=el('div','stat');
    s.append(el('div','sv',typeof v==='number'&&!Number.isInteger(v)?fmt(v):String(v)), el('div','sk',k));
    row.appendChild(s);
  });
  return row;
}
function diamTable(bd){
  const t=el('table','mini');
  const hr=el('tr');
  ['diameter','n','exact match','verdict'].forEach((h,i)=>{
    const th=el('th',i?'num':null,h); hr.appendChild(th);
  });
  const thead=el('thead'); thead.appendChild(hr); t.appendChild(thead);
  const tb=el('tbody');
  Object.entries(bd).forEach(([d,b])=>{
    const tr=el('tr');
    [d, b.n, fmt(b.em), fmt(b.verdict)].forEach((v,i)=>tr.appendChild(el('td',i?'num':null,v)));
    // inline magnitude cue for em: a short single-hue bar under the number
    const bar=el('div','embar'); bar.style.width=Math.round(b.em*56)+'px';
    tr.children[2].appendChild(bar);
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  return t;
}

function openModal(r){
  modal.textContent='';
  const m=r.meta, cfg=m.config||{};
  const x=el('span','x','×'); x.onclick=closeModal;
  const h=el('h3',null,r.id);
  const sub=el('div','tags',`${r.dataset} · ${r.task} · ${r.arch}`);
  modal.append(x,h,sub);
  const add=n=>{ if(n) modal.appendChild(n); };

  // result — the numbers first
  const res={ 'best test': r.best, 'at epoch': r.best_epoch };
  if(m.best_eval) Object.assign(res, {
    'verdict': m.best_eval.verdict, 'p within': m.best_eval.p_within,
    'p cross': m.best_eval.p_cross, 'fp cross': m.best_eval.fp_cross,
    'sparsity': m.best_eval.bdh_sparsity,
  });
  add(section('result (best epoch)', statRow(res)));

  if(m.ood) add(section(`out-of-distribution — ${m.ood.dataset} (n=${m.ood.n})`,
    statRow({ 'exact match': m.ood.em, 'verdict': m.ood.verdict,
              'p within': m.ood.p_within, 'p cross': m.ood.p_cross })));

  if(m.diameter_breakdown)
    add(section('exact match by graph diameter', diamTable(m.diameter_breakdown)));

  // architecture / features / training — grouped config
  add(section('architecture', kvGrid([
    ['architecture', r.arch], ['engine (class)', cfg.model], ['layers', layersLine(cfg.layers)],
    ['hidden', cfg.hidden_channels], ['parameters', r.params],
    ['in → out', cfg.in_channels!=null?`${cfg.in_channels} → ${cfg.out_channels}`:null],
    ['tokenization', cfg.tokenization], ['pooling', cfg.task==='graph'?cfg.pooling:null],
    ['input embedding', cfg.input_embedding], ['norm / residual',
      cfg.norm_type!=null||cfg.residual!=null?`${cfg.norm_type??'none'} / ${cfg.residual?'yes':'no'}`:null],
    ['CoT', cfg.cot_len>0?`${cfg.cot_len} tokens (${cfg.cot_mask})`:null],
    ['BDH neuron mult', cfg.model==='bdh'?cfg.bdh_neuron_mult:null],
    ['BDH adj mask', cfg.model==='bdh'?cfg.bdh_adj_mask:null],
  ])));
  add(section('features & data', kvGrid([
    ['node features', cfg.node_features], ['LPE dim', cfg.lpe_dim||null],
    ['node id dim', cfg.node_id_dim||null], ['dataset', cfg.dataset||r.dataset],
    ['dataset kwargs', cfg.dataset_kwargs&&Object.keys(cfg.dataset_kwargs).length?cfg.dataset_kwargs:null],
    ['seed', cfg.seed], ['train frac', cfg.train_frac],
  ])));
  add(section('training', kvGrid([
    ['epochs', cfg.epochs], ['lr', cfg.lr], ['weight decay', cfg.weight_decay],
    ['batch size', cfg.batch_size], ['dropout', cfg.dropout],
  ])));

  const t=m.timing||{}, inf=t.inference||{};
  add(section('timing', kvGrid([
    ['device', t.device], ['total train', t.total_train_s!=null?`${t.total_train_s} s`:null],
    ['mean epoch', t.mean_epoch_train_s!=null?`${t.mean_epoch_train_s} s`:null],
    ['inference / graph', inf.per_graph_ms!=null?`${inf.per_graph_ms} ms`:null],
    ['inference / call', inf.per_call_ms!=null?`${inf.per_call_ms} ms`:null],
  ])));

  if(m.provenance){ const p=m.provenance;
    add(section('provenance', kvGrid([
      ['git', p.git_commit!=null?`${p.git_commit}${p.git_dirty?' (dirty)':''}`:null],
      ['command', p.argv], ['torch', p.torch], ['python', p.python],
    ])));
  }

  // full raw JSON stays reachable, out of the way
  const det=el('details');
  det.append(el('summary',null,'raw JSON'),
             (()=>{ const pre=el('pre'); pre.textContent=JSON.stringify(m,null,2); return pre; })());
  modal.appendChild(det);

  modalBg.classList.add('show');
}
function closeModal(){ modalBg.classList.remove('show'); }
modalBg.onclick=e=>{ if(e.target===modalBg) closeModal(); };
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModal(); });

// ── canvas helpers ────────────────────────────────────────────────────────────
function setup(canvas,h){
  // a .grow canvas is sized by CSS flex — read its laid-out height; otherwise use h
  const dpr=window.devicePixelRatio||1, w=canvas.clientWidth||600;
  if(canvas.classList.contains('grow') && canvas.clientHeight>0) h=canvas.clientHeight;
  canvas.width=w*dpr; canvas.height=h*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  return {ctx,w,h};
}
function frame(ctx,w,h,pad,yMin,yMax,yTicks,fmtY){
  ctx.strokeStyle=CHROME.grid; ctx.fillStyle=CHROME.muted; ctx.font="10px system-ui";
  ctx.lineWidth=1; ctx.textAlign="right"; ctx.textBaseline="middle";
  for(let k=0;k<=yTicks;k++){
    const v=yMin+(yMax-yMin)*k/yTicks;
    const y=pad.t+(h-pad.t-pad.b)*(1-k/yTicks);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
    ctx.fillText(fmtY(v),pad.l-6,y);
  }
  ctx.strokeStyle=CHROME.axis;
  ctx.beginPath(); ctx.moveTo(pad.l,h-pad.b); ctx.lineTo(w-pad.r,h-pad.b); ctx.stroke();
}
function ema(arr,a){
  if(!a) return arr;
  let m=null; return arr.map(v=>{ if(v==null) return null;
    m = m==null ? v : a*m+(1-a)*v; return m; });
}

// ── line chart (curves view) ──────────────────────────────────────────────────
function drawLines(canvas, field, yMin, yMax, ref, hoverEp){
  const {ctx,w,h}=setup(canvas,300);
  ctx.clearRect(0,0,w,h);
  const pad={l:44,r:14,t:10,b:24};
  const act=active().filter(r=>r.series[field]);
  const maxEp=Math.max(1,...act.map(r=>Math.max(...r.ep)));
  const X=e=>pad.l+(w-pad.l-pad.r)*(e/maxEp);
  const Y=v=>pad.t+(h-pad.t-pad.b)*(1-(v-yMin)/(yMax-yMin));
  frame(ctx,w,h,pad,yMin,yMax,4,v=>v.toFixed(2));
  ctx.fillStyle=CHROME.muted; ctx.textAlign="center"; ctx.textBaseline="top";
  for(let k=0;k<=4;k++){ const e=Math.round(maxEp*k/4); ctx.fillText(e,X(e),h-pad.b+6); }
  if(ref!=null && ref>=yMin && ref<=yMax){         // labeled threshold, not grid
    ctx.strokeStyle=CHROME.axis; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,Y(ref)); ctx.lineTo(w-pad.r,Y(ref)); ctx.stroke();
    ctx.setLineDash([]);
  }
  act.forEach(r=>{
    const ys=ema(r.series[field],smooth);
    ctx.strokeStyle=SLOTS[r.slot]; ctx.lineWidth=2; ctx.beginPath();
    let started=false;
    r.ep.forEach((e,i)=>{ const v=ys[i]; if(v==null) return;
      const x=X(e), y=Y(Math.max(yMin,Math.min(yMax,v)));
      started ? ctx.lineTo(x,y) : ctx.moveTo(x,y); started=true; });
    ctx.stroke();
    if(field===metric && r.best_epoch!=null){       // best-epoch marker
      const i=r.ep.indexOf(r.best_epoch);
      if(i>=0 && r.series[field][i]!=null){
        ctx.fillStyle=SLOTS[r.slot];
        ctx.beginPath(); ctx.arc(X(r.ep[i]),Y(r.series[field][i]),4,0,7); ctx.fill();
        ctx.strokeStyle="#1a1a19"; ctx.lineWidth=2; ctx.stroke();  // 2px surface ring
      }
    }
  });
  if(hoverEp!=null){                                 // crosshair
    ctx.strokeStyle=CHROME.muted; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(X(hoverEp),pad.t); ctx.lineTo(X(hoverEp),h-pad.b); ctx.stroke();
  }
  canvas._hit={pad,w,maxEp,field,act};
}

function lineHover(ev){
  const canvas=ev.currentTarget, hit=canvas._hit; if(!hit||!hit.act.length) return;
  const rect=canvas.getBoundingClientRect();
  const fx=(ev.clientX-rect.left-hit.pad.l)/(rect.width-hit.pad.l-hit.pad.r);
  const epv=Math.round(hit.maxEp*Math.max(0,Math.min(1,fx)));
  drawLines(canvas, hit.field, canvas._ymin, canvas._ymax, canvas._ref, epv);
  tip.textContent='';
  const head=document.createElement('div'); head.className='row';
  head.innerHTML=`epoch <span class="v">${epv}</span>`;
  tip.appendChild(head);
  hit.act.forEach(r=>{
    let bi=-1,bd=1e9;
    r.ep.forEach((e,i)=>{ const d=Math.abs(e-epv); if(d<bd){bd=d;bi=i;} });
    const v=(r.series[hit.field]||[])[bi];
    const row=document.createElement('div'); row.className='row';
    const k=document.createElement('span'); k.className='k'; k.style.borderColor=SLOTS[r.slot];
    const val=document.createElement('span'); val.className='v'; val.textContent=fmt(v);
    row.append(k,val,document.createTextNode('  '+short(r)));
    tip.appendChild(row);
  });
  tip.style.left=Math.min(ev.clientX+14, innerWidth-360)+'px';
  tip.style.top=(ev.clientY+14)+'px'; tip.style.opacity=1;
}
function lineLeave(ev){
  const c=ev.currentTarget, hit=c._hit;
  tip.style.opacity=0;
  if(hit) drawLines(c, hit.field, c._ymin, c._ymax, c._ref, null);
}

// ── scatter: best vs depth, colored by layer type ─────────────────────────────
function drawScatter(canvas, hoverIdx){
  const {ctx,w,h}=setup(canvas,320);
  ctx.clearRect(0,0,w,h);
  const pad={l:44,r:14,t:10,b:26};
  // selected runs when any are selected, else all filtered (never empty by default)
  const sel=active().filter(r=>r.best!=null && r.depth>0);
  const fs=(sel.length ? sel : filtered().filter(r=>r.best!=null && r.depth>0));
  const bySlot=sel.length>0;
  const maxD=Math.max(2,...fs.map(r=>r.depth));
  const X=d=>pad.l+(w-pad.l-pad.r)*(d/(maxD+0.5));
  const Y=v=>pad.t+(h-pad.t-pad.b)*(1-v);
  frame(ctx,w,h,pad,0,1,4,v=>v.toFixed(2));
  ctx.fillStyle=CHROME.muted; ctx.textAlign="center"; ctx.textBaseline="top";
  for(let d=1;d<=maxD;d++) ctx.fillText(d,X(d),h-pad.b+6);
  ctx.strokeStyle=CHROME.axis; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(pad.l,Y(0.5)); ctx.lineTo(w-pad.r,Y(0.5)); ctx.stroke();
  ctx.setLineDash([]);
  // deterministic jitter per run id so overlapping depths separate
  const pts=fs.map((r,i)=>{
    let hsh=0; for(const ch of r.id) hsh=(hsh*31+ch.charCodeAt(0))|0;
    const jx=((hsh%1000)/1000-0.5)*0.3;
    return {r, x:X(r.depth+jx), y:Y(r.best)};
  });
  pts.forEach((p,i)=>{
    ctx.fillStyle = bySlot ? SLOTS[p.r.slot] : layerColor(p.r.layer_type);
    ctx.beginPath(); ctx.arc(p.x,p.y, i===hoverIdx?6:4.5, 0,7); ctx.fill();
    ctx.strokeStyle="#1a1a19"; ctx.lineWidth=2; ctx.stroke();   // 2px surface ring
  });
  canvas._pts=pts;
}
function scatterHover(ev){
  const canvas=ev.currentTarget, pts=canvas._pts; if(!pts) return;
  const rect=canvas.getBoundingClientRect();
  const mx=ev.clientX-rect.left, my=ev.clientY-rect.top;
  let bi=-1,bd=1e9;
  pts.forEach((p,i)=>{ const d=(p.x-mx)**2+(p.y-my)**2; if(d<bd){bd=d;bi=i;} });
  if(bi<0 || bd>24*24){ tip.style.opacity=0; drawScatter(canvas,null); return; }  // nearest-point, 24px
  drawScatter(canvas,bi);
  const r=pts[bi].r;
  tip.textContent='';
  const v=document.createElement('div'); v.className='row';
  v.innerHTML=`<span class="v">${fmt(r.best)}</span> best · depth ${r.depth}`;
  const n=document.createElement('div'); n.className='row'; n.textContent=short(r);
  const t=document.createElement('div'); t.className='row';
  t.textContent=`${r.dataset} · ${r.depth}×${r.layer_type} · h${r.hidden}`;
  tip.append(v,n,t);
  tip.style.left=Math.min(ev.clientX+14, innerWidth-360)+'px';
  tip.style.top=(ev.clientY+14)+'px'; tip.style.opacity=1;
}

// ── EM vs graph diameter (from each run's diameter_breakdown) ─────────────────
function diamRuns(){
  // selected runs with a breakdown; if none selected has one, fall back to the
  // first 8 filtered runs that do (colored by layer type)
  const sel=active().filter(r=>r.by_diam);
  if(sel.length) return {rs:sel, bySlot:true};
  return {rs:filtered().filter(r=>r.by_diam).slice(0,8), bySlot:false};
}
function drawDiam(canvas, hoverD){
  const {ctx,w,h}=setup(canvas,300);
  ctx.clearRect(0,0,w,h);
  const pad={l:44,r:14,t:10,b:26};
  const {rs,bySlot}=diamRuns();
  canvas._diam={rs,bySlot,pad,w};
  if(!rs.length){
    ctx.fillStyle=CHROME.muted; ctx.font="12px system-ui"; ctx.textAlign="center";
    ctx.fillText("no diameter breakdown recorded — only runs made after the logging upgrade have it",
                 w/2, h/2);
    return;
  }
  const maxD=Math.max(2,...rs.flatMap(r=>Object.keys(r.by_diam).map(Number)));
  const X=d=>pad.l+(w-pad.l-pad.r)*(d/(maxD+0.5));
  const Y=v=>pad.t+(h-pad.t-pad.b)*(1-v);
  frame(ctx,w,h,pad,0,1,4,v=>v.toFixed(2));
  ctx.fillStyle=CHROME.muted; ctx.textAlign="center"; ctx.textBaseline="top";
  for(let d=1;d<=maxD;d++) ctx.fillText(d,X(d),h-pad.b+6);
  rs.forEach(r=>{
    const col=bySlot?SLOTS[r.slot]:layerColor(r.layer_type);
    const ds=Object.keys(r.by_diam).map(Number).sort((a,b)=>a-b);
    ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath();
    ds.forEach((d,i)=>{ const y=Y(r.by_diam[d].em);
      i ? ctx.lineTo(X(d),y) : ctx.moveTo(X(d),y); });
    ctx.stroke();
    ds.forEach(d=>{
      ctx.fillStyle=col;
      ctx.beginPath(); ctx.arc(X(d),Y(r.by_diam[d].em),3.5,0,7); ctx.fill();
      ctx.strokeStyle="#1a1a19"; ctx.lineWidth=2; ctx.stroke();
    });
  });
  if(hoverD!=null){
    ctx.strokeStyle=CHROME.muted; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(X(hoverD),pad.t); ctx.lineTo(X(hoverD),h-pad.b); ctx.stroke();
  }
  canvas._diam.X=X; canvas._diam.maxD=maxD;
}
function diamHover(ev){
  const canvas=ev.currentTarget, st=canvas._diam;
  if(!st || !st.rs.length) return;
  const rect=canvas.getBoundingClientRect();
  const fx=(ev.clientX-rect.left-st.pad.l)/(rect.width-st.pad.l-st.pad.r);
  const d=Math.round((st.maxD+0.5)*Math.max(0,Math.min(1,fx)));
  drawDiam(canvas,d);
  tip.textContent='';
  const head=document.createElement('div'); head.className='row';
  head.innerHTML=`diameter <span class="v">${d}</span>`;
  tip.appendChild(head);
  st.rs.forEach(r=>{
    const b=r.by_diam[d]; if(!b) return;
    const row=document.createElement('div'); row.className='row';
    const k=document.createElement('span'); k.className='k';
    k.style.borderColor=st.bySlot?SLOTS[r.slot]:layerColor(r.layer_type);
    const val=document.createElement('span'); val.className='v';
    val.textContent=`em ${fmt(b.em)} · verdict ${fmt(b.verdict)}`;
    row.append(k,val,document.createTextNode(` (n=${b.n})  ${short(r)}`));
    tip.appendChild(row);
  });
  tip.style.left=Math.min(ev.clientX+14, innerWidth-360)+'px';
  tip.style.top=(ev.clientY+14)+'px'; tip.style.opacity=1;
}

// ── ranked bars: best per run (single series -> slot 1) ───────────────────────
function drawBars(canvas, hoverIdx){
  const fs=filtered().filter(r=>r.best!=null)
    .sort((a,b)=>b.best-a.best).slice(0,30);
  const rowH=22, pad={l:8,r:60,t:4,b:4};
  const h=pad.t+pad.b+fs.length*rowH;
  const {ctx,w}=setup(canvas,h);
  ctx.clearRect(0,0,w,h);
  const X=v=>pad.l+(w-pad.l-pad.r)*v;
  const bars=fs.map((r,i)=>({r, y:pad.t+i*rowH}));
  bars.forEach((b,i)=>{
    const bh=12, y=b.y+(rowH-bh)/2, x1=X(Math.max(0,b.r.best));
    ctx.fillStyle = i===hoverIdx ? "#5598e7" : (b.r.slot!=null ? SLOTS[b.r.slot] : "#3987e5");
    ctx.beginPath(); ctx.roundRect(pad.l,y,Math.max(2,x1-pad.l),bh,[0,4,4,0]); ctx.fill();
    ctx.fillStyle=CHROME.ink2; ctx.font="10px system-ui";
    ctx.textAlign="left"; ctx.textBaseline="middle";
    ctx.fillText(fmt(b.r.best), x1+6, b.y+rowH/2);
    const label=short(b.r);
    ctx.fillStyle="#0d0d0d"; ctx.font="10px system-ui";
    const fits = ctx.measureText(label).width < (x1-pad.l-12);
    if(fits) ctx.fillText(label, pad.l+6, b.y+rowH/2);   // in-bar only when it fits
    else { ctx.fillStyle=CHROME.muted;
           ctx.fillText(label.slice(0,40), x1+50, b.y+rowH/2); }
  });
  canvas._bars={bars,rowH};
}
function barsHover(ev){
  const canvas=ev.currentTarget, bb=canvas._bars; if(!bb) return;
  const rect=canvas.getBoundingClientRect();
  const idx=Math.floor((ev.clientY-rect.top-4)/bb.rowH);
  if(idx<0||idx>=bb.bars.length){ tip.style.opacity=0; drawBars(canvas,null); return; }
  drawBars(canvas,idx);
  const r=bb.bars[idx].r;
  tip.textContent='';
  const v=document.createElement('div'); v.className='row';
  v.innerHTML=`<span class="v">${fmt(r.best)}</span> best @ epoch ${r.best_epoch ?? '–'}`;
  const n=document.createElement('div'); n.className='row'; n.textContent=r.id;
  tip.append(v,n);
  tip.style.left=Math.min(ev.clientX+14, innerWidth-360)+'px';
  tip.style.top=(ev.clientY+14)+'px'; tip.style.opacity=1;
}

// ── generic XY scatter (Overview grid) ────────────────────────────────────────
// spec: { x, y (accessors, null skips a run), xmin?, xmax?, xlog?, ymin=0, ymax=1,
//         xfmt?, yfmt?, xticks?, diag?, hline?, empty?, tip(r)->[strings] }
function compactNum(v){
  return v>=1e6 ? (v/1e6).toFixed(1)+'M' : v>=1e3 ? Math.round(v/1e3)+'k' : String(v);
}
function drawXY(canvas, spec){
  const {ctx,w,h}=setup(canvas, 300);
  ctx.clearRect(0,0,w,h);
  const pad={l:46,r:16,t:12,b:30};
  const rs=spec.runs.filter(r=>spec.x(r)!=null && spec.y(r)!=null);
  canvas._xy={spec,rs,pad,w,h,pts:[]};
  if(!rs.length){
    ctx.fillStyle=CHROME.muted; ctx.font="12px system-ui"; ctx.textAlign="center";
    ctx.fillText(spec.empty||"no runs with this data yet", w/2, h/2);
    return;
  }
  const xlog=!!spec.xlog;
  const tx=v=> xlog ? Math.log10(Math.max(v,1)) : v;
  let xs=rs.map(r=>tx(spec.x(r)));
  let xmin=spec.xmin!=null?tx(spec.xmin):Math.min(...xs);
  let xmax=spec.xmax!=null?tx(spec.xmax):Math.max(...xs);
  if(xmin===xmax){ xmin-=0.5; xmax+=0.5; }
  else { const m=(xmax-xmin)*0.05; xmin-=m; xmax+=m; }
  const ymin=spec.ymin??0, ymax=spec.ymax??1;
  const X=v=>pad.l+(w-pad.l-pad.r)*((tx(v)-xmin)/(xmax-xmin));
  const Y=v=>pad.t+(h-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));
  frame(ctx,w,h,pad,ymin,ymax,4,spec.yfmt||(v=>v.toFixed(2)));
  // x ticks
  ctx.fillStyle=CHROME.muted; ctx.textAlign="center"; ctx.textBaseline="top";
  ctx.font="10px system-ui";
  let ticks=spec.xticks;
  if(!ticks){
    ticks=[];
    if(xlog){ for(let p=Math.ceil(xmin);p<=Math.floor(xmax);p++) ticks.push(Math.pow(10,p)); }
    else { for(let k=0;k<=4;k++) ticks.push(spec.x==null?0:(xmin+(xmax-xmin)*k/4)); }
    if(!ticks.length) ticks=[Math.pow(10,xmin),Math.pow(10,xmax)];
  }
  const xf=spec.xfmt||(xlog?compactNum:(v=>(+v).toFixed(2)));
  ticks.forEach(vx=>{ const px=X(vx);
    if(px>=pad.l-1 && px<=w-pad.r+1) ctx.fillText(xf(vx),px,h-pad.b+6); });
  if(spec.xlabel){ ctx.textAlign="right"; ctx.textBaseline="bottom";
    ctx.fillText(spec.xlabel, w-pad.r, h-1); }
  // diagonal y=x (assumes shared 0..1 axes)
  if(spec.diag){ ctx.strokeStyle=CHROME.axis; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(X(0),Y(0)); ctx.lineTo(X(1),Y(1)); ctx.stroke(); ctx.setLineDash([]); }
  if(spec.hline!=null){ ctx.strokeStyle=CHROME.axis; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,Y(spec.hline)); ctx.lineTo(w-pad.r,Y(spec.hline)); ctx.stroke(); ctx.setLineDash([]); }
  // points, colored by layer type; selected runs get a white outline ring
  const pts=rs.map(r=>({r, x:X(spec.x(r)), y:Y(spec.y(r))}));
  pts.forEach(p=>{
    ctx.fillStyle=layerColor(p.r.layer_type);
    ctx.beginPath(); ctx.arc(p.x,p.y,4.5,0,7); ctx.fill();
    ctx.strokeStyle = p.r.slot!=null ? "#fff" : "#1a1a19";
    ctx.lineWidth=2; ctx.stroke();
  });
  canvas._xy.pts=pts; canvas._xy.hover=null;
}
function drawXYHover(canvas, idx){
  const st=canvas._xy; if(!st) return;
  // redraw base + emphasized point
  drawXY(canvas, st.spec);
  const p=canvas._xy.pts[idx]; if(!p) return;
  const {ctx}=setup(canvas,300);
  ctx.fillStyle=layerColor(p.r.layer_type);
  ctx.beginPath(); ctx.arc(p.x,p.y,6.5,0,7); ctx.fill();
  ctx.strokeStyle="#1a1a19"; ctx.lineWidth=2; ctx.stroke();
}
function xyHover(ev){
  const canvas=ev.currentTarget, st=canvas._xy; if(!st||!st.pts.length) return;
  const rect=canvas.getBoundingClientRect();
  const sx=(canvas.width/(window.devicePixelRatio||1))/rect.width;
  const mx=(ev.clientX-rect.left)*sx, my=(ev.clientY-rect.top)*sx;
  let bi=-1,bd=1e9;
  st.pts.forEach((p,i)=>{ const d=(p.x-mx)**2+(p.y-my)**2; if(d<bd){bd=d;bi=i;} });
  if(bi<0 || bd>26*26){ tip.style.opacity=0; drawXY(canvas,st.spec); return; }
  drawXYHover(canvas,bi);
  const r=st.pts[bi].r;
  tip.textContent='';
  const n=document.createElement('div'); n.className='row'; n.textContent=short(r);
  tip.appendChild(n);
  (st.spec.tip?st.spec.tip(r):[]).forEach(line=>{
    const row=document.createElement('div'); row.className='row'; row.textContent=line;
    tip.appendChild(row);
  });
  tip.style.left=Math.min(ev.clientX+14, innerWidth-360)+'px';
  tip.style.top=(ev.clientY+14)+'px'; tip.style.opacity=1;
}
function bindXY(canvas, spec){
  drawXY(canvas, spec);
  canvas.addEventListener('mousemove', xyHover);
  canvas.addEventListener('mouseleave', ()=>{ tip.style.opacity=0; drawXY(canvas, canvas._xy.spec); });
  canvas.addEventListener('click', ev=>{
    const st=canvas._xy; if(!st||!st.pts.length) return;
    const rect=canvas.getBoundingClientRect();
    const sx=(canvas.width/(window.devicePixelRatio||1))/rect.width;
    const mx=(ev.clientX-rect.left)*sx, my=(ev.clientY-rect.top)*sx;
    let bi=-1,bd=1e9;
    st.pts.forEach((p,i)=>{ const d=(p.x-mx)**2+(p.y-my)**2; if(d<bd){bd=d;bi=i;} });
    if(bi>=0 && bd<=26*26){ activate(st.pts[bi].r); renderAll(); }
  });
}

// ── views ─────────────────────────────────────────────────────────────────────
function card(title,hint){
  const c=document.createElement('div'); c.className='card';
  const h=document.createElement('h2'); h.textContent=title;
  const hi=document.createElement('div'); hi.className='hint'; hi.textContent=hint;
  c.append(h,hi); return c;
}
function canvasIn(c,id,cls){ const cv=document.createElement('canvas'); cv.id=id;
  if(cls) cv.className=cls; c.appendChild(cv); return cv; }
function layerLegend(runs){
  const leg=document.createElement('div'); leg.className='legend';
  const present=new Set(runs.map(r=>r.layer_type));
  LAYER_ORDER.filter(t=>present.has(t)).forEach(t=>{
    const it=document.createElement('span'); it.className='it';
    const d=document.createElement('span'); d.className='dot'; d.style.background=layerColor(t);
    it.append(d,document.createTextNode(t)); leg.appendChild(it);
  });
  return leg;
}

function viewOverview(main){
  const fs=filtered();
  const charts=document.createElement('div'); charts.className='charts grid2 fill';

  // 1. capacity — accuracy vs parameter count (log x)
  const c1=card('Accuracy vs parameters',
    'each run · color = layer type · x = trainable params (log) · click a point to select · dashed = chance');
  c1.appendChild(layerLegend(fs.filter(r=>r.params!=null && r.best!=null)));
  const a1=canvasIn(c1,'ovA','grow');

  // 2. generalization — in-distribution vs OOD (only connectivity runs log OOD)
  const c2=card('In-distribution vs OOD exact-match',
    'x = test EM · y = OOD (ER) EM · dashed = y=x · points below the line lost accuracy off-distribution');
  const a2=canvasIn(c2,'ovB','grow');

  // 3. failure mode — the all-ones collapse map
  const c3=card('All-ones collapse map',
    'x = P(reachable | same component) · y = P(reachable | different) · top-right = collapsed to all-ones · below 0.5 line = components separated');
  const a3=canvasIn(c3,'ovC','grow');

  // 4. the thesis scatter — best metric vs depth
  const c4=card('Best metric vs depth',
    'x = number of layers/rounds · color = layer type · dashed = chance');
  c4.appendChild(layerLegend(fs.filter(r=>r.best!=null && r.depth>0)));
  const a4=canvasIn(c4,'ovD','grow');

  charts.append(c1,c2,c3,c4);
  main.appendChild(charts);

  bindXY(a1, { runs:fs, x:r=>r.params, y:r=>r.best, xlog:true, ymin:0, ymax:1,
    hline:0.5, xlabel:'params',
    empty:'no runs log a parameter count yet',
    tip:r=>[`best ${fmt(r.best)} · ${r.params?.toLocaleString()} params`,
            `${r.dataset} · ${r.depth}×${r.layer_type} · h${r.hidden}`] });

  bindXY(a2, { runs:fs, x:r=>r.best, y:r=>r.ood, xmin:0, xmax:1, ymin:0, ymax:1,
    diag:true, xlabel:'test EM',
    empty:'no runs with an OOD probe yet (connectivity runs after the logging upgrade)',
    tip:r=>[`test ${fmt(r.best)} → OOD ${fmt(r.ood)}`,
            `${r.dataset} · ${r.depth}×${r.layer_type}`] });

  bindXY(a3, { runs:fs, x:r=>r.p_within, y:r=>r.p_cross, xmin:0, xmax:1, ymin:0, ymax:1,
    hline:0.5, xlabel:'P(reachable | same comp)',
    empty:'no runs with within/cross probabilities yet (connectivity runs)',
    tip:r=>[`p_within ${fmt(r.p_within)} · p_cross ${fmt(r.p_cross)}`,
            `EM ${fmt(r.best)} · ${r.depth}×${r.layer_type}`] });

  const maxD=Math.max(2,...fs.filter(r=>r.depth>0).map(r=>r.depth));
  bindXY(a4, { runs:fs.filter(r=>r.best!=null && r.depth>0),
    x:r=>r.depth, y:r=>r.best, xmin:0.5, xmax:maxD+0.5, ymin:0, ymax:1,
    hline:0.5, xlabel:'depth', xticks:Array.from({length:maxD},(_,i)=>i+1),
    xfmt:v=>String(Math.round(v)),
    tip:r=>[`best ${fmt(r.best)} · depth ${r.depth}`,
            `${r.dataset} · ${r.depth}×${r.layer_type} · h${r.hidden}`] });
}

function viewCurves(main){
  const act=active();
  if(!act.length){
    const e=document.createElement('div'); e.className='empty';
    e.textContent='Select runs in the sidebar (up to 8) to plot their training curves.';
    main.appendChild(e); return;
  }
  const charts=document.createElement('div'); charts.className='charts grid2 fill';
  // 1. selected metric (with the metric picker + smoothing control)
  const c1=card('', '');
  c1.textContent='';
  const h=document.createElement('h2');
  h.append('Metric: ');
  const sel=document.createElement('select');
  metricList().forEach(m=>{ const o=document.createElement('option'); o.value=m; o.textContent=m; sel.appendChild(o); });
  sel.value=metric; sel.onchange=()=>{ metric=sel.value; renderMain(); };
  const sl=document.createElement('label'); sl.className='sl';
  sl.append(' smoothing ');
  const rng=document.createElement('input'); rng.type='range'; rng.min=0; rng.max=0.9; rng.step=0.1; rng.value=smooth;
  rng.oninput=()=>{ smooth=+rng.value; redrawCurves(); };
  sl.appendChild(rng);
  h.append(sel,sl);
  const hint=document.createElement('div'); hint.className='hint';
  hint.textContent='higher is better · dashed = 0.5 chance · dot = best epoch · hover for values';
  c1.append(h,hint);
  const mC=canvasIn(c1,'mC','grow');
  // 2. loss  3. gradient norm  4. epoch time
  const c2=card('Training loss','BCE / cross-entropy · dashed = ln 2 (uninformed 50/50)');
  const lC=canvasIn(c2,'lC','grow');
  const c3=card('Gradient norm','global L2 of parameter grads per epoch · autoscaled · only runs logged after the upgrade');
  const gC=canvasIn(c3,'gC','grow');
  const c4=card('Epoch time (s)','per-epoch training wall time · inference latency in each run’s ⓘ');
  const tC=canvasIn(c4,'tC','grow');
  charts.append(c1,c2,c3,c4);
  main.appendChild(charts);
  redrawCurves();
  [mC,lC,gC,tC].forEach(cv=>{
    cv.addEventListener('mousemove',lineHover);
    cv.addEventListener('mouseleave',lineLeave);
  });
}
function redrawCurves(){
  const mC=$('mC'), lC=$('lC'), gC=$('gC'), tC=$('tC');
  if(!mC) return;
  const act=active();
  const seriesMax=(k,floor)=>{ let m=floor;
    act.forEach(r=>(r.series[k]||[]).forEach(v=>{ if(v!=null&&v>m)m=v; })); return m; };
  const lmax=seriesMax('loss',0.8), tmax=seriesMax('train_time_s',0.05), gmax=seriesMax('grad_norm',0);
  // metrics in [0,1] get a fixed scale + chance line; unbounded ones autoscale
  const bounded = seriesMax(metric,0)<=1;
  mC._ymin=0; mC._ymax=bounded?1:seriesMax(metric,0)*1.05; mC._ref=bounded?0.5:null;
  lC._ymin=0; lC._ymax=Math.min(lmax,1.2); lC._ref=0.6931;
  gC._ymin=0; gC._ymax=Math.max(gmax*1.1,1e-6); gC._ref=null;
  tC._ymin=0; tC._ymax=tmax*1.1; tC._ref=null;
  drawLines(mC,metric,0,mC._ymax,mC._ref,null);
  drawLines(lC,'loss',0,lC._ymax,0.6931,null);
  drawLines(gC,'grad_norm',0,gC._ymax,null,null);
  drawLines(tC,'train_time_s',0,tC._ymax,null,null);
}

function viewCompare(main){
  const charts=document.createElement('div'); charts.className='charts';
  const anySel=active().some(r=>r.best!=null && r.depth>0);
  const c1=card('Best metric vs depth', anySel
    ? 'selected runs only · color = slot · dashed = chance · clear the selection to see all runs'
    : 'all filtered runs · color = layer type · select runs in the sidebar to narrow to them · dashed = chance');
  if(!anySel) c1.appendChild(layerLegend(filtered()));  // layer legend only when coloring by layer
  const sC=canvasIn(c1,'sC');
  const cd=card('Exact-match vs graph diameter',
    'the depth↔reachability curve · selected runs (or first 8 with data) · hover a diameter for em / verdict / n');
  const dC=canvasIn(cd,'dC');
  const c2=card('Ranking — best metric per run','top 30 of the filtered runs · selected runs keep their slot color');
  const bC=canvasIn(c2,'bC');
  charts.append(c1,cd,c2);
  main.appendChild(charts);
  drawScatter(sC,null); drawDiam(dC,null); drawBars(bC,null);
  dC.addEventListener('mousemove',diamHover);
  dC.addEventListener('mouseleave',()=>{ tip.style.opacity=0; drawDiam(dC,null); });
  sC.addEventListener('mousemove',scatterHover);
  sC.addEventListener('mouseleave',()=>{ tip.style.opacity=0; drawScatter(sC,null); });
  bC.addEventListener('mousemove',barsHover);
  bC.addEventListener('mouseleave',()=>{ tip.style.opacity=0; drawBars(bC,null); });
  bC.addEventListener('click',ev=>{
    const bb=bC._bars; if(!bb) return;
    const rect=bC.getBoundingClientRect();
    const idx=Math.floor((ev.clientY-rect.top-4)/bb.rowH);
    if(idx>=0&&idx<bb.bars.length){ activate(bb.bars[idx].r); renderAll(); }
  });
}

const COLS=[
  ["run","id",false],["dataset","dataset",false],["arch","arch",false],
  ["layers",null,false],["h","hidden",true],["params","params",true],["lr","lr",true],
  ["ep","epochs",true],
  ["best","best",true],["best@","best_epoch",true],["ood em","ood",true],
  ["ms/ep","ms_epoch",true],["infer ms","infer_ms",true],
];
function viewTable(main){
  const fs=filtered().slice();
  const key=sortKey;
  fs.sort((a,b)=>{
    const va = key==="layers" ? a.depth : a[key], vb = key==="layers" ? b.depth : b[key];
    if(va==null) return 1; if(vb==null) return -1;
    return (va<vb?-1:va>vb?1:0)*sortDir;
  });
  const tbl=document.createElement('table');
  const thead=document.createElement('thead'); const trh=document.createElement('tr');
  COLS.forEach(([label,k,num])=>{
    const th=document.createElement('th'); if(num) th.className='num';
    const kk = k ?? "layers";
    th.textContent=label + (sortKey===kk ? (sortDir<0?' ↓':' ↑') : '');
    th.onclick=()=>{ if(sortKey===kk) sortDir*=-1; else { sortKey=kk; sortDir=-1; } renderMain(); };
    trh.appendChild(th);
  });
  thead.appendChild(trh); tbl.appendChild(thead);
  const tb=document.createElement('tbody');
  fs.forEach(r=>{
    const tr=document.createElement('tr');
    if(r.slot!=null){ tr.className='on'; tr.style.setProperty('--rowc',SLOTS[r.slot]); }
    const cells=[short(r),r.dataset,r.arch,`${r.depth}×${r.layer_type}`,r.hidden,
                 r.params!=null?r.params.toLocaleString():'–',r.lr,r.epochs,
                 fmt(r.best),r.best_epoch??'–',r.ood!=null?fmt(r.ood):'–',
                 r.ms_epoch??'–',r.infer_ms??'–'];
    cells.forEach((v,i)=>{ const td=document.createElement('td');
      if(COLS[i][2]) td.className='num'; td.textContent=v; tr.appendChild(td); });
    tr.onclick=()=>{ activate(r); renderAll(); };
    tr.ondblclick=()=>openModal(r);
    tb.appendChild(tr);
  });
  tbl.appendChild(tb);
  main.appendChild(tbl);
}

const VIEW_HINT={
  overview:"every run at a glance · click a point to select · filters scope all four charts",
  curves:"select up to 8 runs in the sidebar to plot their training curves",
  compare:"click a point or bar to select a run · filters scope every chart",
  table:"click a row to select · double-click for full details",
};
function renderMain(){
  const main=$('main'); main.textContent='';
  $('hint').textContent=VIEW_HINT[view]||'';
  if(view==='overview') viewOverview(main);
  else if(view==='curves') viewCurves(main);
  else if(view==='compare') viewCompare(main);
  else viewTable(main);
}
function renderAll(){ renderList(); renderMain(); }

// ── wiring ────────────────────────────────────────────────────────────────────
$('tabs').querySelectorAll('button').forEach(b=>{
  b.onclick=()=>{ view=b.dataset.v;
    $('tabs').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));
    renderMain(); };
});
['fDs','fTask','fLayer','fModel'].forEach(id=>$(id).onchange=renderAll);
$('fQ').oninput=renderAll;
$('clear').onclick=()=>{ RUNS.forEach(r=>r.slot=null); renderAll(); };
$('top8').onclick=()=>{
  RUNS.forEach(r=>r.slot=null);
  filtered().filter(r=>r.best!=null).sort((a,b)=>b.best-a.best)
    .slice(0,8).forEach((r,i)=>r.slot=i);
  renderAll();
};
window.addEventListener('resize',()=>{ if(view!=='table') renderMain(); });

// initial state from the URL hash, e.g. report.html#view=compare&top8=1&dataset=er
(function(){
  const p=new URLSearchParams(location.hash.slice(1));
  if(p.get('dataset')) $('fDs').value=p.get('dataset');
  if(p.get('view')){ view=p.get('view');
    $('tabs').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.v===view)); }
  if(p.get('top8')) filtered().filter(r=>r.best!=null)
      .sort((a,b)=>b.best-a.best).slice(0,8).forEach((r,i)=>r.slot=i);
  if(p.get('modal')){
    const r=RUNS.find(x=>x.id.includes(p.get('modal')));
    if(r) openModal(r);
  }
})();
renderAll();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="directory of run JSONs")
    ap.add_argument("--out", default=None, help="output HTML path (default: figures/report.html)")
    ap.add_argument("--open", action="store_true", help="open the report in a browser")
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        print(f"No result JSONs found in {args.results}/")
        return

    out = args.out or os.path.join("figures", "report.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(build_html(runs))
    print(f"Wrote {out}  ({len(runs)} runs)")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
