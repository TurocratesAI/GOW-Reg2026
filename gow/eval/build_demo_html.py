#!/usr/bin/env python3
"""Build the self-contained pathologist demo page from gow/artifacts/demo_data.json."""
import json, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "..", "artifacts", "demo_data.json")))
DATA_JSON = json.dumps(data)

PAGE = r"""
<title>Grounded Ontology Walker - reasoning demo</title>
<style>
:root{
  --bg:#F1F3F1; --panel:#FFFFFF; --panel-2:#F7F8F6; --ink:#1C2A2E; --muted:#5E6C71;
  --line:#E1E6E2; --accent:#0E6A6E; --accent-weak:#0e6a6e18; --accent-ink:#0b4f52;
  --pos:#2f7d5b; --neg:#8a5a2b; --shadow:0 1px 2px #1c2a2e0d,0 8px 24px -12px #1c2a2e22;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#0E1418; --panel:#151D22; --panel-2:#111a1f; --ink:#E7EDEA; --muted:#90A0A0;
    --line:#243039; --accent:#45C8BE; --accent-weak:#45c8be1f; --accent-ink:#8fe6de;
    --pos:#67c79a; --neg:#d1a06a; --shadow:0 1px 2px #0007,0 10px 30px -14px #000a; }
}
:root[data-theme="light"]{ --bg:#F1F3F1; --panel:#FFFFFF; --panel-2:#F7F8F6; --ink:#1C2A2E; --muted:#5E6C71;
  --line:#E1E6E2; --accent:#0E6A6E; --accent-weak:#0e6a6e18; --accent-ink:#0b4f52; --pos:#2f7d5b; --neg:#8a5a2b;
  --shadow:0 1px 2px #1c2a2e0d,0 8px 24px -12px #1c2a2e22; }
:root[data-theme="dark"]{ --bg:#0E1418; --panel:#151D22; --panel-2:#111a1f; --ink:#E7EDEA; --muted:#90A0A0;
  --line:#243039; --accent:#45C8BE; --accent-weak:#45c8be1f; --accent-ink:#8fe6de; --pos:#67c79a; --neg:#d1a06a;
  --shadow:0 1px 2px #0007,0 10px 30px -14px #000a; }

*{box-sizing:border-box}
body{margin:0}
.wrap{ font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); background:var(--bg);
  min-height:100vh; padding:22px clamp(14px,3vw,34px) 40px; line-height:1.5; }
.top{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap;
  padding-bottom:16px; border-bottom:1px solid var(--line); margin-bottom:20px; }
.brand{ display:flex; gap:12px; align-items:center; }
.logo{ width:34px;height:34px;border-radius:9px; display:grid;place-items:center; color:#fff;
  background:linear-gradient(135deg,var(--accent),var(--accent-ink)); font-size:19px; box-shadow:var(--shadow); }
.title{ font-size:19px; font-weight:680; letter-spacing:-.01em; }
.sub{ font-size:12.5px; color:var(--muted); }
.case{ display:flex; gap:7px; flex-wrap:wrap; }
.chip{ font-size:12px; padding:4px 10px; border:1px solid var(--line); border-radius:999px; background:var(--panel);
  color:var(--muted); font-variant-numeric:tabular-nums; }
.chip b{ color:var(--ink); font-weight:600; }

.grid{ display:grid; grid-template-columns:minmax(0,1.05fr) minmax(320px,.95fr); gap:20px; align-items:start; }
@media (max-width:880px){ .grid{ grid-template-columns:1fr; } }
.panel{ background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:var(--shadow); }
.panel-head{ display:flex; justify-content:space-between; align-items:center; padding:13px 16px; border-bottom:1px solid var(--line); }
.label{ font-size:11px; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); font-weight:640; }

.viewer{ position:relative; margin:14px; border-radius:10px; overflow:hidden; border:1px solid var(--line);
  background:var(--panel-2); }
.viewer img{ display:block; width:100%; height:auto; }
.viewer .fade{ transition:opacity .18s ease; }
.viewer-cap{ padding:2px 16px 4px; font-size:13px; color:var(--muted); min-height:20px; }
.viewer-cap b{ color:var(--ink); }
.legend{ display:flex; align-items:center; gap:9px; padding:8px 16px 16px; font-size:11px; color:var(--muted);
  letter-spacing:.03em; }
.legend .bar{ flex:1; height:8px; border-radius:5px;
  background:linear-gradient(90deg,#000010,#4a0c6b,#b6377a,#f98e09,#fcffa4); border:1px solid var(--line); }
.switch{ display:inline-flex; align-items:center; gap:7px; font-size:12.5px; color:var(--muted); cursor:pointer; user-select:none; }
.switch input{ accent-color:var(--accent); width:15px; height:15px; }

.body{ padding:16px; }
.gen{ width:100%; font:inherit; font-weight:640; font-size:15px; color:#fff; cursor:pointer;
  background:var(--accent); border:0; border-radius:11px; padding:13px 16px; letter-spacing:.01em;
  box-shadow:var(--shadow); transition:filter .15s ease, transform .05s ease; }
.gen:hover{ filter:brightness(1.06); } .gen:active{ transform:translateY(1px); }
.gen:disabled{ opacity:.5; cursor:default; }
.hint{ margin-top:12px; font-size:13px; color:var(--muted); }

.chain{ display:flex; flex-direction:column; gap:5px; margin-top:14px; }
.step{ display:grid; grid-template-columns:1fr auto; gap:8px 12px; align-items:baseline; text-align:left;
  width:100%; font:inherit; cursor:pointer; background:transparent; border:1px solid transparent;
  border-left:3px solid transparent; border-radius:9px; padding:9px 12px; color:var(--ink);
  transition:background .13s ease,border-color .13s ease; }
.step:hover{ background:var(--panel-2); }
.step.active{ background:var(--accent-weak); border-color:var(--line); border-left-color:var(--accent); }
.step .q{ font-size:13.5px; color:var(--muted); }
.step.active .q{ color:var(--ink); }
.step .a{ font-size:13.5px; font-weight:620; color:var(--accent-ink); text-align:right; font-variant-numeric:tabular-nums; }
.step .look{ grid-column:1/-1; font-size:11px; color:var(--accent); letter-spacing:.02em; margin-top:2px;
  opacity:0; height:0; overflow:hidden; transition:opacity .13s ease; }
.step.active .look{ opacity:1; height:auto; }

.report{ margin-top:18px; border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.report .rh{ padding:9px 15px; border-bottom:1px solid var(--line); background:var(--panel-2);
  font-size:11px; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); font-weight:640;
  display:flex; justify-content:space-between; }
.report .rb{ padding:15px 17px; font-family:Georgia,"Times New Roman",serif; font-size:15.5px; line-height:1.65;
  white-space:pre-line; color:var(--ink); }
.reveal{ opacity:0; transform:translateY(6px); animation:rise .34s ease forwards; }
@keyframes rise{ to{ opacity:1; transform:none; } }
@media (prefers-reduced-motion:reduce){ .reveal{ animation:none; opacity:1; transform:none; } .viewer .fade{ transition:none; } }
.foot{ margin-top:20px; font-size:12px; color:var(--muted); text-align:center; }
.foot b{ color:var(--ink); }
</style>

<div class="wrap">
  <header class="top">
    <div class="brand">
      <span class="logo">&#9681;</span>
      <div>
        <div class="title">Grounded Ontology Walker</div>
        <div class="sub">Diagnostic reasoning, grounded on the slide</div>
      </div>
    </div>
    <div class="case" id="caseChips"></div>
  </header>

  <main class="grid">
    <section class="panel">
      <div class="panel-head">
        <span class="label">Whole-slide image</span>
        <label class="switch"><input type="checkbox" id="attnToggle" checked> Attention overlay</label>
      </div>
      <div class="viewer"><img id="slideImg" class="fade" alt="whole-slide image"></div>
      <div class="viewer-cap" id="cap">H&amp;E thumbnail. Generate the report, then select any answer to see where the model looked.</div>
      <div class="legend"><span>low attention</span><div class="bar"></div><span>high</span></div>
    </section>

    <section class="panel">
      <div class="panel-head"><span class="label">Reasoning</span><span class="label" id="stepCount"></span></div>
      <div class="body">
        <button id="genBtn" class="gen">Generate report</button>
        <p class="hint" id="hint">Runs the ontology walk over the slide: at each question the model answers from the
          tissue it attends to, and the transition table supplies the next question, until it renders the report.</p>
        <div id="chain" class="chain" hidden></div>
        <div id="report" class="report" hidden></div>
      </div>
    </section>
  </main>
  <div class="foot">Attention shown is the model&#39;s question-conditioned pooling weight, learned only from
    case-level answers. Illustrative research demo &middot; <b>not for diagnostic use</b>.</div>
</div>

<script>
const DATA = /*__DATA__*/;
const THUMB = "data:image/png;base64," + DATA.thumbnail;
const img = document.getElementById("slideImg");
const cap = document.getElementById("cap");
const chainEl = document.getElementById("chain");
const reportEl = document.getElementById("report");
const attn = document.getElementById("attnToggle");
let sel = -1;

img.src = THUMB;
document.getElementById("caseChips").innerHTML =
  `<span class="chip"><b>${cap0(DATA.organ)}</b></span>` +
  `<span class="chip">${DATA.steps[1] ? DATA.steps[1].answer : "biopsy"}</span>` +
  `<span class="chip"><b>${DATA.n_tiles.toLocaleString()}</b> tiles @20x</span>`;

function cap0(s){ return s ? s[0].toUpperCase()+s.slice(1) : s; }

function paint(){
  if(sel>=0 && attn.checked){ img.style.opacity=0;
    setTimeout(()=>{ img.src = "data:image/png;base64,"+DATA.steps[sel].heatmap; img.style.opacity=1; }, 120); }
  else { img.style.opacity=0; setTimeout(()=>{ img.src = THUMB; img.style.opacity=1; }, 120); }
  cap.innerHTML = (sel>=0)
    ? `Attention for &mdash; <b>${escapeHtml(DATA.steps[sel].question)}</b>`
    : "H&amp;E thumbnail. Select any answer to see where the model looked.";
}
function select(i){
  sel = (sel===i) ? -1 : i;
  [...chainEl.children].forEach((el,k)=>el.classList.toggle("active", k===sel));
  paint();
}
attn.addEventListener("change", paint);

document.getElementById("genBtn").addEventListener("click", function(){
  this.disabled = true; this.textContent = "Report generated";
  document.getElementById("hint").hidden = true;
  document.getElementById("stepCount").textContent = DATA.steps.length + " reasoning steps";
  chainEl.hidden = false; chainEl.innerHTML = "";
  DATA.steps.forEach((s,i)=>{
    const b = document.createElement("button");
    b.className = "step reveal"; b.style.animationDelay = (i*45)+"ms";
    b.innerHTML = `<span class="q">${escapeHtml(s.question)}</span><span class="a">${escapeHtml(s.answer)}</span>`+
                  `<span class="look">&#9681; show where the model looked</span>`;
    b.addEventListener("click", ()=>select(i));
    chainEl.appendChild(b);
  });
  reportEl.hidden = false;
  reportEl.className = "report reveal"; reportEl.style.animationDelay = (DATA.steps.length*45+80)+"ms";
  reportEl.innerHTML = `<div class="rh"><span>Pathology report</span><span>rendered from the chain</span></div>`+
                       `<div class="rb">${escapeHtml(DATA.report)}</div>`;
});

function escapeHtml(s){ return (s||"").replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
</script>
"""

out = PAGE.replace("/*__DATA__*/", DATA_JSON)
dst = os.path.join(HERE, "..", "..", "paper", "demo", "index.html")
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write(out)
print(f"[demo] wrote {dst}  ({len(out)//1024} KB)")
