"""Result visualizer — turns results/*.json into one self-contained HTML dashboard.

Scans every run, embeds the data inline (no server, no CDN, works offline), and
renders interactive loss / test-accuracy curves you can toggle and compare.

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
    """Read every run JSON into a compact record the dashboard can plot."""
    runs = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json") or fname == "report.html":
            continue
        with open(os.path.join(results_dir, fname)) as f:
            run = json.load(f)
        cfg = run.get("config", {})
        hist = run.get("history", [])
        summary = run.get("summary", {})
        layers = cfg.get("layers", [])
        runs.append({
            "id": run.get("run_id", fname),
            "dataset": run.get("dataset", "?"),
            "tokenization": cfg.get("tokenization", "node"),
            "depth": len(layers),
            "layer_type": layers[0]["type"] if layers else "?",
            "hidden": cfg.get("hidden_channels", "-"),
            "lr": cfg.get("lr", "-"),
            "epochs": cfg.get("epochs", len(hist)),
            "best": summary.get("test", summary.get("val", None)),
            "best_epoch": summary.get("best_epoch", None),
            # series kept parallel + short to keep the HTML small
            "ep": [h["epoch"] for h in hist],
            "loss": [h.get("loss") for h in hist],
            "acc": [h.get("test", h.get("val")) for h in hist],
        })
    return runs


def build_html(runs: list[dict]) -> str:
    data_json = json.dumps(runs, separators=(",", ":"))
    return _TEMPLATE.replace("/*DATA*/", data_json).replace("/*COUNT*/", str(len(runs)))


# ── The dashboard: a single HTML string with embedded canvas charts ─────────────
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mpGNN results</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --txt:#e6e9ef;
    --muted:#8b93a7; --accent:#4f9cff;
  }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--txt); }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex;
           align-items:baseline; gap:14px; }
  header h1 { font-size:16px; margin:0; font-weight:650; }
  header .sub { color:var(--muted); font-size:12px; }
  .wrap { display:flex; height:calc(100vh - 51px); }
  .side { width:360px; min-width:300px; border-right:1px solid var(--line);
          overflow-y:auto; padding:12px; }
  .main { flex:1; padding:16px 20px; overflow-y:auto; }
  .controls { display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
  select, button { background:var(--panel); color:var(--txt); border:1px solid var(--line);
                   border-radius:6px; padding:5px 9px; font-size:12px; cursor:pointer; }
  button:hover, select:hover { border-color:var(--accent); }
  .run { display:flex; align-items:center; gap:8px; padding:7px 8px; border-radius:7px;
         cursor:pointer; border:1px solid transparent; }
  .run:hover { background:var(--panel); }
  .run.on { background:var(--panel); border-color:var(--line); }
  .sw { width:11px; height:11px; border-radius:3px; flex:none; }
  .run .meta { flex:1; min-width:0; }
  .run .name { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:12px; }
  .run .tags { color:var(--muted); font-size:11px; }
  .run .best { font-variant-numeric:tabular-nums; font-weight:650; font-size:12px; }
  .charts { display:grid; grid-template-columns:1fr; gap:18px; max-width:980px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .card h2 { margin:0 0 4px; font-size:13px; font-weight:600; }
  .card .hint { color:var(--muted); font-size:11px; margin-bottom:6px; }
  canvas { width:100%; display:block; }
  .empty { color:var(--muted); padding:40px; text-align:center; }
  .tip { position:fixed; pointer-events:none; background:#000; border:1px solid var(--line);
         border-radius:6px; padding:5px 8px; font-size:11px; opacity:0; transition:opacity .08s;
         white-space:nowrap; z-index:10; }
</style>
</head>
<body>
<header>
  <h1>mpGNN results</h1>
  <span class="sub"><span id="cnt">/*COUNT*/</span> runs · click to toggle · hover for values</span>
</header>
<div class="wrap">
  <aside class="side">
    <div class="controls">
      <select id="ds"><option value="">all datasets</option></select>
      <button id="all">select all</button>
      <button id="none">clear</button>
    </div>
    <div id="list"></div>
  </aside>
  <main class="main">
    <div class="charts">
      <div class="card">
        <h2>Test accuracy</h2>
        <div class="hint">higher is better · 0.5 = chance on a balanced binary task</div>
        <canvas id="accC"></canvas>
      </div>
      <div class="card">
        <h2>Training loss</h2>
        <div class="hint">cross-entropy · 0.693 = ln 2 = predicting 50/50 for everything</div>
        <canvas id="lossC"></canvas>
      </div>
    </div>
  </main>
</div>
<div class="tip" id="tip"></div>

<script>
const RUNS = /*DATA*/;
const PALETTE = ["#4f9cff","#ff6b6b","#51cf66","#fab005","#cc5de8","#22b8cf",
                 "#ff922b","#94d82d","#f06595","#845ef7","#20c997","#fcc419"];
RUNS.forEach((r,i)=>{ r.color = PALETTE[i % PALETTE.length]; r.on = true; });

const tip = document.getElementById('tip');

function fmt(v){ return v==null ? "–" : (Math.round(v*1000)/1000).toFixed(3); }

// ── sidebar ──────────────────────────────────────────────────────────────────
const dsSel = document.getElementById('ds');
[...new Set(RUNS.map(r=>r.dataset))].sort().forEach(d=>{
  const o=document.createElement('option'); o.value=d; o.textContent=d; dsSel.appendChild(o);
});

function shortName(r){
  // drop the leading timestamp for readability
  return r.id.replace(/^\d{8}_\d{6}_/, '');
}

function renderList(){
  const list=document.getElementById('list'); list.innerHTML='';
  const filt=dsSel.value;
  RUNS.forEach(r=>{
    if(filt && r.dataset!==filt) return;
    const el=document.createElement('div');
    el.className='run'+(r.on?' on':'');
    el.innerHTML=`<span class="sw" style="background:${r.on?r.color:'#3a4150'}"></span>
      <div class="meta">
        <div class="name" title="${r.id}">${shortName(r)}</div>
        <div class="tags">${r.tokenization} · depth ${r.depth} · ${r.layer_type} · lr ${r.lr} · ${r.epochs}ep</div>
      </div>
      <div class="best" style="color:${r.best>=0.9?'#51cf66':r.best>=0.7?'#fab005':'#ff6b6b'}">${fmt(r.best)}</div>`;
    el.onclick=()=>{ r.on=!r.on; renderList(); drawAll(); };
    list.appendChild(el);
  });
}

// ── charts ───────────────────────────────────────────────────────────────────
function setup(canvas){
  const dpr=window.devicePixelRatio||1;
  const w=canvas.clientWidth, h=320;
  canvas.width=w*dpr; canvas.height=h*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  return {ctx,w,h};
}

function draw(canvas, field, yMin, yMax, refLine){
  const {ctx,w,h}=setup(canvas);
  ctx.clearRect(0,0,w,h);
  const pad={l:42,r:14,t:10,b:26};
  const active=RUNS.filter(r=>r.on && r[field] && r[field].length);
  const maxEp=Math.max(1,...active.map(r=>Math.max(...r.ep)));
  const X=e=>pad.l+(w-pad.l-pad.r)*(e/maxEp);
  const Y=v=>pad.t+(h-pad.t-pad.b)*(1-(v-yMin)/(yMax-yMin));

  // grid + y labels
  ctx.strokeStyle="#262b36"; ctx.fillStyle="#8b93a7"; ctx.font="10px sans-serif";
  ctx.textAlign="right"; ctx.textBaseline="middle"; ctx.lineWidth=1;
  for(let k=0;k<=4;k++){
    const v=yMin+(yMax-yMin)*k/4, y=Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
    ctx.fillText(v.toFixed(2),pad.l-6,y);
  }
  // x labels
  ctx.textAlign="center"; ctx.textBaseline="top";
  for(let k=0;k<=4;k++){
    const e=Math.round(maxEp*k/4);
    ctx.fillText(e,X(e),h-pad.b+6);
  }
  // reference line (chance / ln2)
  if(refLine!=null && refLine>=yMin && refLine<=yMax){
    ctx.strokeStyle="#4a5160"; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,Y(refLine)); ctx.lineTo(w-pad.r,Y(refLine)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // series
  active.forEach(r=>{
    ctx.strokeStyle=r.color; ctx.lineWidth=1.6; ctx.beginPath();
    let started=false;
    r.ep.forEach((e,i)=>{
      const v=r[field][i]; if(v==null) return;
      const x=X(e), y=Y(v);
      started ? ctx.lineTo(x,y) : ctx.moveTo(x,y); started=true;
    });
    ctx.stroke();
  });

  // hover
  canvas._hit={X,Y,pad,w,h,maxEp,field,active};
}

function drawAll(){
  draw(document.getElementById('accC'),'acc',0.4,1.0,0.5);
  draw(document.getElementById('lossC'),'loss',0.0,Math.max(0.8,lossMax()),0.6931);
}
function lossMax(){
  let m=0.8;
  RUNS.filter(r=>r.on).forEach(r=>r.loss.forEach(v=>{ if(v!=null&&v>m)m=v; }));
  return Math.min(m,1.2);
}

// hover tooltip: nearest epoch across active runs
function onMove(ev){
  const canvas=ev.currentTarget, hit=canvas._hit; if(!hit) return;
  const rect=canvas.getBoundingClientRect();
  const mx=ev.clientX-rect.left;
  const ep=Math.round(hit.maxEp*(mx-hit.pad.l)/(hit.w-hit.pad.l-hit.pad.r));
  if(ep<0||ep>hit.maxEp){ tip.style.opacity=0; return; }
  let lines=[`epoch ${ep}`];
  hit.active.forEach(r=>{
    let bi=-1,bd=1e9;
    r.ep.forEach((e,i)=>{ const d=Math.abs(e-ep); if(d<bd){bd=d;bi=i;} });
    if(bi>=0) lines.push(`<span style="color:${r.color}">●</span> ${fmt(r[hit.field][bi])}`);
  });
  tip.innerHTML=lines.join("<br>");
  tip.style.left=(ev.clientX+14)+'px'; tip.style.top=(ev.clientY+14)+'px';
  tip.style.opacity=1;
}
['accC','lossC'].forEach(id=>{
  const c=document.getElementById(id);
  c.addEventListener('mousemove',onMove);
  c.addEventListener('mouseleave',()=>tip.style.opacity=0);
});

// ── controls ─────────────────────────────────────────────────────────────────
dsSel.onchange=()=>{ renderList(); drawAll(); };
document.getElementById('all').onclick=()=>{
  const filt=dsSel.value;
  RUNS.forEach(r=>{ if(!filt||r.dataset===filt) r.on=true; });
  renderList(); drawAll();
};
document.getElementById('none').onclick=()=>{
  RUNS.forEach(r=>r.on=false); renderList(); drawAll();
};
window.addEventListener('resize',drawAll);

renderList(); drawAll();
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
