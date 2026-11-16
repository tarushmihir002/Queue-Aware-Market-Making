#!/usr/bin/env python
"""
Build a claude.ai-artifact-ready HTML dashboard (platform supplies the
<head>/<body> skeleton) from a pipeline's dashboard_data.json. Equity curves are
downsampled to keep the embedded payload small.

    python scripts/make_artifact.py --data results/run/dashboard_data.json --out results/run/artifact_dashboard.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _downsample(seq, n=500):
    if len(seq) <= n:
        return seq
    step = len(seq) / n
    return [seq[int(i * step)] for i in range(n)]


def prepare(data: dict) -> dict:
    for name, s in data.get("equity", {}).items():
        idx = _downsample(list(range(len(s["t"]))))
        s["t"] = [round(s["t"][i], 2) for i in idx]
        s["pnl"] = [round(s["pnl"][i], 4) for i in idx]
        s["inventory"] = [round(s["inventory"][i], 4) for i in idx]
    return data


def build(data: dict) -> str:
    return _TEMPLATE.replace("/*__DATA__*/", json.dumps(prepare(data)))


_TEMPLATE = r"""<title>Queue-Aware Market Making</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap">
<style>
/* Layout: single column, summary KPIs first, tabbed detail; cards only where an object needs setting off.
   Light-first tokens; accent = market-maker teal; semantic green/red for P&L. */
:root{
 --bg:#f5f8fc;--panel:#ffffff;--panel2:#eef3fa;--ink:#131b2d;--muted:#596a86;--line:#dde4f0;
 --accent:#0e9f70;--green:#0e9f70;--red:#d94759;--amber:#bf8521;--violet:#6f52c9;--blue:#2d77d4;
 --fd:"Space Grotesk",system-ui,sans-serif;--fb:"Inter",system-ui,sans-serif;color-scheme:light;}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#0e1421;--panel:#161f30;--panel2:#1c273b;--ink:#e9eef7;--muted:#95a3be;--line:#2a3651;
 --accent:#33d49c;--green:#33d49c;--red:#f0616f;--amber:#e3a74b;--violet:#9d7fe6;--blue:#5ca3f2;color-scheme:dark;}}
:root[data-theme=dark]{
 --bg:#0e1421;--panel:#161f30;--panel2:#1c273b;--ink:#e9eef7;--muted:#95a3be;--line:#2a3651;
 --accent:#33d49c;--green:#33d49c;--red:#f0616f;--amber:#e3a74b;--violet:#9d7fe6;--blue:#5ca3f2;color-scheme:dark;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--fb);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));
 padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:26px 72px}
h1{font-family:var(--fd);font-weight:700;font-size:26px;letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}
h2{font-family:var(--fd);font-weight:600;font-size:16px;margin:0 0 12px}
.sub{color:var(--muted);margin:0 0 16px;max-width:70ch}
.badges{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:20px}
.badge{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:3px 9px;color:var(--muted);
 font-size:12px;font-variant-numeric:tabular-nums}
.badge b{color:var(--ink);font-weight:600}
.grid{display:grid;gap:13px}
.g4{grid-template-columns:repeat(auto-fit,minmax(168px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:17px;min-width:0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px 16px}
.kpi .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-family:var(--fd);font-weight:600;font-size:23px;margin-top:3px;font-variant-numeric:tabular-nums}
.kpi .d{font-size:12px;margin-top:3px;color:var(--muted);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
tr.best td{background:color-mix(in srgb,var(--accent) 13%,transparent)}
tr.best td:first-child{font-weight:600}
.pos{color:var(--green)}.neg{color:var(--red)}
.tabs{display:flex;gap:7px;flex-wrap:wrap;margin:22px 0 13px}
.tab{padding:7px 14px;border:1px solid var(--line);border-radius:999px;cursor:pointer;color:var(--muted);
 background:var(--panel);font-size:13px;font-weight:500}
.tab:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tab[aria-selected=true]{color:#fff;background:var(--accent);border-color:var(--accent);font-weight:600}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]) .tab[aria-selected=true]{color:#07140d}}
:root[data-theme=dark] .tab[aria-selected=true]{color:#07140d}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:9px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.note{color:var(--muted);font-size:12px;margin-top:10px}
.scroll{overflow-x:auto}
svg{display:block;width:100%;height:auto}
.gl{stroke:var(--line);stroke-dasharray:2 3;stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:10px;font-family:var(--fb)}
.foot{color:var(--muted);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:14px}
</style>
<div class="wrap">
<h1>Queue-Aware Market Making</h1>
<p class="sub" id="sub"></p>
<div class="badges" id="badges"></div>
<div class="grid g4" id="kpis"></div>
<div class="tabs" id="tabs" role="tablist"></div>
<div id="panels"></div>
<p class="foot">Hawkes order flow · survival-analysis fill probability · adverse-selection markouts ·
Avellaneda–Stoikov vs Bayesian vs RL. Objective = spread revenue − adverse selection − fees − inventory cost.
P&amp;L in quote currency; markouts in ticks.</p>
</div>
<script>
const DATA=/*__DATA__*/;
const PAL={join_touch:"#2d77d4",fixed_spread_1t:"#bf8521",avellaneda_stoikov:"#d94759",
 bayesian_queue:"#0e9f70",rl_dqn:"#6f52c9"};
const SER=["#2d77d4","#0e9f70","#bf8521","#6f52c9","#d94759","#1fa8a8"];
function el(t,a={},h){const e=document.createElement(t);for(const k in a)e.setAttribute(k,a[k]);if(h!=null)e.innerHTML=h;return e;}
function fmt(x,d=3){return(x==null||Number.isNaN(x))?"–":(+x).toFixed(d);}
function sign(x){return x>0?"pos":x<0?"neg":"";}
function nice(s){return s.replace(/_/g," ").replace(/\b1t\b/,"(1 tick)");}

function lineChart(series,{h=290,xlab="",zero=true}={}){
 const W=560,H=h,pad={l:54,r:14,t:12,b:30};
 let xs=[],ys=[];series.forEach(s=>s.x.forEach((v,i)=>{xs.push(v);ys.push(s.y[i]);}));
 let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 if(zero){ymin=Math.min(ymin,0);ymax=Math.max(ymax,0);}if(ymin===ymax){ymax+=1;ymin-=1;}
 const pdY=(ymax-ymin)*.06;ymin-=pdY;ymax+=pdY;
 const X=v=>pad.l+(v-xmin)/((xmax-xmin)||1)*(W-pad.l-pad.r);
 const Y=v=>H-pad.b-(v-ymin)/((ymax-ymin)||1)*(H-pad.t-pad.b);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
 for(let i=0;i<=4;i++){const yv=ymin+(ymax-ymin)*i/4,yy=Y(yv).toFixed(1);
  s+=`<line class="gl" x1="${pad.l}" y1="${yy}" x2="${W-pad.r}" y2="${yy}"/>`;
  s+=`<text class="tick" x="${pad.l-6}" y="${(+yy+3)}" text-anchor="end">${(+yv).toPrecision(3)}</text>`;}
 for(let i=0;i<=4;i++){const xv=xmin+(xmax-xmin)*i/4;
  s+=`<text class="tick" x="${X(xv)}" y="${H-pad.b+15}" text-anchor="middle">${(+xv).toFixed(0)}</text>`;}
 if(zero&&ymin<0&&ymax>0){const y0=Y(0).toFixed(1);s+=`<line class="axis" x1="${pad.l}" y1="${y0}" x2="${W-pad.r}" y2="${y0}"/>`;}
 series.forEach(ser=>{let d="";ser.x.forEach((v,i)=>{d+=(i?"L":"M")+X(v).toFixed(1)+" "+Y(ser.y[i]).toFixed(1)+" ";});
  s+=`<path d="${d}" fill="none" stroke="${ser.color}" stroke-width="1.8" stroke-linejoin="round"/>`;});
 s+=`<text class="tick" x="${W/2}" y="${H-3}" text-anchor="middle">${xlab}</text></svg>`;return s;
}
function barChart(cats,vals,colors,{h=290}={}){
 const W=560,H=h,pad={l:52,r:14,t:12,b:72};
 let ymin=Math.min(0,...vals),ymax=Math.max(0,...vals);if(ymin===ymax)ymax+=1;
 const pd=(ymax-ymin)*.08;ymax+=pd;if(ymin<0)ymin-=pd;
 const Y=v=>H-pad.b-(v-ymin)/((ymax-ymin)||1)*(H-pad.t-pad.b);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
 for(let i=0;i<=4;i++){const yv=ymin+(ymax-ymin)*i/4,yy=Y(yv).toFixed(1);
  s+=`<line class="gl" x1="${pad.l}" y1="${yy}" x2="${W-pad.r}" y2="${yy}"/>`;
  s+=`<text class="tick" x="${pad.l-6}" y="${(+yy+3)}" text-anchor="end">${(+yv).toPrecision(2)}</text>`;}
 const y0=Y(0).toFixed(1);s+=`<line class="axis" x1="${pad.l}" y1="${y0}" x2="${W-pad.r}" y2="${y0}"/>`;
 const bw=(W-pad.l-pad.r)/cats.length*0.62;
 cats.forEach((c,i)=>{const cx=pad.l+(i+0.5)/cats.length*(W-pad.l-pad.r),yy=Y(vals[i]);
  const col=Array.isArray(colors)?colors[i]:colors;
  s+=`<rect x="${(cx-bw/2).toFixed(1)}" y="${Math.min(yy,y0).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.abs(yy-y0).toFixed(1)}" rx="3" fill="${col}"/>`;
  s+=`<text class="tick" x="${cx}" y="${H-pad.b+15}" text-anchor="end" transform="rotate(-30 ${cx} ${H-pad.b+15})">${c}</text>`;});
 return s+"</svg>";
}
function legend(series){return `<div class="legend">`+series.map(s=>`<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")+`</div>`;}

// header
const t=DATA.data.test;
document.getElementById("sub").textContent=
 `${DATA.symbol} · tick ${DATA.tick_size} · out-of-sample test ${(t.duration_s/60).toFixed(0)} min · `+
 `${t.n_book_updates.toLocaleString()} book updates, ${t.n_trades.toLocaleString()} trades. `+
 `The execution-aware Bayesian quoter is compared against classical and RL baselines.`;
const hk=DATA.hawkes||{},q=DATA.queue;
[["branching ratio",fmt(hk.branching_ratio,2)],["Hawkes MLE fit br",fmt(hk.fit_branching_ratio,2)],
 ["κ decay /tick",fmt(DATA.kappa,2)],["queue rate bid",fmt(q.rate_bid,1)+"/s"],
 ["markout R² @1s",fmt(DATA.markout_r2["1.0"],2)]].forEach(([l,v])=>{
 const b=el("span",{class:"badge"});b.innerHTML=`${l} <b>${v}</b>`;document.getElementById("badges").appendChild(b);});

const strat=DATA.strategies;
const best=strat.reduce((a,b)=>b.objective>a.objective?b:a);
const bq=strat.find(s=>s.strategy==="bayesian_queue"),as_=strat.find(s=>s.strategy==="avellaneda_stoikov");
const kp=document.getElementById("kpis");
function kpi(l,v,d,cls){const e=el("div",{class:"kpi"});e.appendChild(el("div",{class:"l"},l));
 e.appendChild(el("div",{class:"v "+(cls||"")},v));if(d!=null)e.appendChild(el("div",{class:"d"},d));kp.appendChild(e);}
kpi("Best by objective",nice(best.strategy),"net P&L "+fmt(best.net_pnl,1));
if(bq)kpi("Bayesian net P&L",fmt(bq.net_pnl,1),"objective "+fmt(bq.objective,1),sign(bq.net_pnl));
if(bq&&as_)kpi("Bayesian − AS objective","+"+fmt(bq.objective-as_.objective,1),"adverse-selection edge",(bq.objective-as_.objective)>0?"pos":"neg");
if(bq)kpi("Bayesian adverse sel.",fmt(bq.adverse_selection_1s,2),"lowest of all strategies");

const tabs=["Strategies","Fill model","Adverse selection","Objective"];
const tabEl=document.getElementById("tabs"),panEl=document.getElementById("panels"),panels=[];
tabs.forEach((label,i)=>{const b=el("button",{class:"tab",role:"tab",id:"tab"+i,"aria-selected":i?"false":"true"},label);
 b.onclick=()=>{document.querySelectorAll(".tab").forEach(x=>x.setAttribute("aria-selected","false"));
  panels.forEach(p=>p.hidden=true);b.setAttribute("aria-selected","true");panels[i].hidden=false;};
 tabEl.appendChild(b);const p=el("div",{role:"tabpanel"});p.hidden=i>0;panEl.appendChild(p);panels.push(p);});

// panel 0 — strategies
(function(){const p=panels[0];
 const cols=[["strategy","Strategy"],["net_pnl","Net P&L",2],["objective","Objective",2],["spread_revenue","Spread",2],
  ["adverse_selection_1s","Adv sel",2],["fees","Fees",2],["inventory_cost","Inv cost",3],["n_fills","Fills",0],
  ["fill_rate_per_quote","Fill/quote",2],["avg_markout_ticks","Markout(t)",3],["mean_abs_inventory","|inv|",1],
  ["sharpe_annualised","Sharpe",0]];
 let h=`<div class="card scroll"><table><thead><tr>`+cols.map(c=>`<th>${c[1]}</th>`).join("")+`</tr></thead><tbody>`;
 strat.forEach(s=>{h+=`<tr class="${s.strategy===best.strategy?'best':''}">`+cols.map(c=>{
  if(c[0]==="strategy")return `<td>${nice(s.strategy)}</td>`;
  const v=s[c[0]],cl=["net_pnl","objective","spread_revenue"].includes(c[0])?sign(v):"";
  return `<td class="${cl}">${c[2]===0?v:fmt(v,c[2])}</td>`;}).join("")+`</tr>`;});
 h+=`</tbody></table></div>`;
 const names=Object.keys(DATA.equity);
 const sEq=names.map(n=>({name:nice(n),x:DATA.equity[n].t,y:DATA.equity[n].pnl,color:PAL[n]||SER[0]}));
 const sInv=names.map(n=>({name:nice(n),x:DATA.equity[n].t,y:DATA.equity[n].inventory,color:PAL[n]||SER[0]}));
 h+=`<div class="grid g2" style="margin-top:13px">
  <div class="card"><h2>Out-of-sample equity (P&L)</h2>${lineChart(sEq,{xlab:"time (s)"})}${legend(sEq)}</div>
  <div class="card"><h2>Inventory trajectory</h2>${lineChart(sInv,{xlab:"time (s)"})}${legend(sInv)}</div></div>`;
 p.innerHTML=h;})();

// panel 1 — fill model
(function(){const p=panels[1];const fe=DATA.fill_eval,taus=Object.keys(fe);
 const rel=taus.map((tt,k)=>({name:`τ=${tt}s (AUC ${fmt(fe[tt].auc,2)})`,x:fe[tt].reliability.map(r=>r[0]),
  y:fe[tt].reliability.map(r=>r[1]),color:SER[k]}));
 rel.push({name:"perfect calibration",x:[0,1],y:[0,1],color:"var(--muted)"});
 const aj=DATA.aj;const sAj=[{name:"filled",x:aj.grid,y:aj.cif_fill,color:"#0e9f70"},
  {name:"requoted / cancelled",x:aj.grid,y:aj.cif_requote,color:"#d94759"},
  {name:"still resting",x:aj.grid,y:aj.surv,color:"#8a97b0"}];
 let h=`<div class="grid g4">`+taus.map(tt=>`<div class="kpi"><div class="l">AUC @ ${tt}s</div>
  <div class="v">${fmt(fe[tt].auc,3)}</div><div class="d">Brier ${fmt(fe[tt].brier,3)} vs ${fmt(fe[tt].brier_const,3)} base</div></div>`).join("")+`</div>`;
 h+=`<div class="grid g2" style="margin-top:13px">
  <div class="card"><h2>Fill-probability calibration</h2>${lineChart(rel,{xlab:"predicted P(fill ≤ τ)",zero:false})}${legend(rel)}
   <p class="note">Points on the diagonal = well-calibrated probabilities.</p></div>
  <div class="card"><h2>Competing risks (Aalen–Johansen)</h2>${lineChart(sAj,{xlab:"time since placement (s)",zero:false})}${legend(sAj)}
   <p class="note">Fill competes with cancellation; the gap is pure adverse-selection exposure.</p></div></div>`;
 p.innerHTML=h;})();

// panel 2 — adverse selection
(function(){const p=panels[2];const touch=DATA.markout_table.filter(r=>r.dist_ticks===0);
 const cats=touch.map(r=>`q ${r.queue_lo}–${r.queue_hi>=10?'∞':r.queue_hi}`);
 let h=`<div class="grid g2">
  <div class="card"><h2>Markout @1s by queue position (at touch)</h2>
   ${barChart(cats,touch.map(r=>r.markout_ticks),touch.map(r=>r.markout_ticks<0?"#d94759":"#0e9f70"))}
   <p class="note">More volume ahead ⇒ slower, more adversely-selected fills.</p></div>
  <div class="card"><h2>Fill rate by queue position</h2>${barChart(cats,touch.map(r=>r.fill_rate),"#2d77d4")}</div></div>`;
 const c2=[["dist_ticks","dist(t)",0],["q","queue"],["n","n",0],["fill_rate","fill rate",2],["p_fill_100ms","P≤100ms",2],
  ["p_fill_500ms","P≤500ms",2],["p_fill_1s","P≤1s",2],["p_requote","P requote",2],["median_fill_time","med fill(s)",2],
  ["markout_ticks","markout(t)",2]];
 h+=`<div class="card scroll" style="margin-top:13px"><table><thead><tr>`+c2.map(c=>`<th>${c[1]}</th>`).join("")+`</tr></thead><tbody>`;
 DATA.markout_table.forEach(r=>{h+=`<tr>`+c2.map(c=>{
  if(c[0]==="q")return `<td>${r.queue_lo}–${r.queue_hi>=10?'∞':r.queue_hi}</td>`;
  const v=r[c[0]],cl=c[0]==="markout_ticks"?sign(v):"";return `<td class="${cl}">${c[2]===0?v:fmt(v,c[2])}</td>`;}).join("")+`</tr>`;});
 h+=`</tbody></table></div>`;p.innerHTML=h;})();

// panel 3 — objective decomposition
(function(){const p=panels[3];const oc=DATA.objective_components;
 const cats=oc.map(o=>nice(o.strategy)),mx=Math.max(...oc.map(o=>o.objective));
 let h=`<div class="card"><h2>Net objective by strategy</h2>
  ${barChart(cats,oc.map(o=>o.objective),oc.map(o=>o.objective>0?"#0e9f70":"#d94759"))}
  <div class="scroll" style="margin-top:12px"><table><thead><tr>`
  +["Strategy","Spread rev","− Adv sel","− Fees","− Inv cost","= Objective","Net P&L"].map(c=>`<th>${c}</th>`).join("")
  +`</tr></thead><tbody>`;
 oc.forEach(o=>{h+=`<tr class="${o.objective===mx?'best':''}"><td>${nice(o.strategy)}</td>
  <td class="pos">${fmt(o.spread_revenue,2)}</td><td class="neg">${fmt(-o.adverse_selection,2)}</td>
  <td class="neg">${fmt(-o.fees,2)}</td><td class="neg">${fmt(-o.inventory_cost,3)}</td>
  <td class="${sign(o.objective)}">${fmt(o.objective,2)}</td><td class="${sign(o.net_pnl)}">${fmt(o.net_pnl,2)}</td></tr>`;});
 h+=`</tbody></table></div><p class="note">The strategies differ most on the adverse-selection term — not on spread —
  which is the project's central thesis.</p></div>`;p.innerHTML=h;})();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    html = build(data)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
