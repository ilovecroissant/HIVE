"use client";
import { useState } from "react";
import type { Module } from "../lib/types";
const sites: {id:string;x:number;y:number;w:number;h:number}[]=[
{id:"hab1",x:235,y:208,w:110,h:60},{id:"solar",x:88,y:258,w:118,h:52},{id:"life",x:396,y:156,w:96,h:55},
{id:"green",x:430,y:302,w:115,h:55},{id:"reactor",x:238,y:355,w:93,h:58},{id:"comms",x:552,y:240,w:65,h:52}
];
export default function MarsMap({modules,crisis}:{modules:Module[];crisis:boolean}){
 const [selected,setSelected]=useState<string|null>("hab1");
 const moduleForSite=(id:string|null)=>modules.find(m=>m.id===id||m.type===({hab1:"habitat",life:"life_support",green:"greenhouse"} as Record<string,string>)[id||""]);
 const current=moduleForSite(selected);
 return <div className="map-wrap">
 <div className="map-overlay"><span className="tiny-label">OLYMPUS MONS • SECTOR 07</span><span className="coordinate">18°39′N 226°12′E</span></div>
 <svg className="mars-map" viewBox="0 0 700 500" role="img" aria-label="Interactive isometric Mars colony layout">
 <defs>
 <linearGradient id="ground" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#422a27"/><stop offset=".55" stopColor="#291e20"/><stop offset="1" stopColor="#162331"/></linearGradient>
 <linearGradient id="dome" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#678899"/><stop offset=".5" stopColor="#304857"/><stop offset="1" stopColor="#172734"/></linearGradient>
 <pattern id="dots" width="25" height="25" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r=".7" fill="#b88674" opacity=".32"/></pattern>
 <filter id="glow"><feGaussianBlur stdDeviation="5"/></filter>
 </defs>
 <path d="M0 80 L310 0 L700 75 L700 500 L0 500Z" fill="url(#ground)"/>
 <path d="M0 80 L310 0 L700 75 L700 500 L0 500Z" fill="url(#dots)"/>
 <g opacity=".22" stroke="#c68c71" strokeWidth="1" fill="none"><path d="M0 162 C180 95 287 175 700 110"/><path d="M0 362 C190 309 431 378 700 290"/><path d="M30 450 C240 402 460 490 680 402"/><path d="M180 0 C120 151 225 340 171 500"/></g>
 <ellipse cx="344" cy="275" rx="265" ry="178" fill="#d37c48" opacity=".035"/>
 <path d="M138 293 L268 232 L443 204 L573 270 L482 335 L289 389 L138 293Z" fill="#a5624d" fillOpacity=".08" stroke="#d28a62" strokeOpacity=".35" strokeWidth="2" strokeDasharray="7 7"/>
 <path d="M288 245 L450 182 M288 245 L150 276 M288 245 L295 372 M288 245 L481 321 M450 182 L580 264" stroke="#f1ad78" strokeOpacity=".28" strokeWidth="9"/>
 <path d="M288 245 L450 182 M288 245 L150 276 M288 245 L295 372 M288 245 L481 321 M450 182 L580 264" stroke="#ffe0a7" strokeOpacity=".35" strokeWidth="2" strokeDasharray="6 7"/>
 {sites.map(s=>{const m=moduleForSite(s.id);const active=selected===s.id;const affected=crisis&&(s.id==="solar"||s.id==="life");return <g key={s.id} role="button" tabIndex={0} aria-label={`Select ${m?.label || s.id}`} onClick={()=>setSelected(s.id)} onKeyDown={e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();setSelected(s.id);}}} className="map-site" transform={`translate(${s.x} ${s.y})`}>
 <ellipse cx={s.w/2} cy={s.h*.82} rx={s.w*.55} ry={s.h*.35} fill="#000" opacity=".35"/>
 {s.id==="solar"?<g><path d="M2 19 L87 0 L118 20 L29 42 Z" fill="#183b58" stroke="#65a1bd" strokeWidth="2"/><path d="M24 16 L51 34 M45 11 L73 30 M66 6 L95 25 M17 27 L104 13" stroke="#7cb5d1" opacity=".6"/><path d="M45 38 L45 51 M81 31 L81 49" stroke="#7a7c80" strokeWidth="4"/></g>:s.id==="comms"?<g><path d="M33 49 L30 6 L14 46" stroke="#9eb4bb" strokeWidth="4" fill="none"/><ellipse cx="30" cy="14" rx="24" ry="11" transform="rotate(-29 30 14)" fill="#365469" stroke="#9bb8c6" strokeWidth="2"/><circle cx="33" cy="13" r="4" fill="#f4ad71"/></g>:s.id==="reactor"?<g><path d="M10 22 L54 2 L88 20 L43 44Z" fill="#365264" stroke="#8bafb8" strokeWidth="2"/><path d="M10 22 V42 L43 58 V44 M43 58 L88 36 V20" fill="#233847" stroke="#7293a6" strokeWidth="2"/><rect x="39" y="21" width="13" height="20" fill="#f6a769" opacity=".8"/></g>:<g><rect x="7" y="32" width={s.w-14} height={s.h-30} rx="7" fill="#304653" stroke="#6d8e9a" strokeWidth="2"/><path d={`M8 33 Q${s.w/2} -23 ${s.w-8} 33 Z`} fill="url(#dome)" stroke="#8db6bf" strokeWidth="2"/><path d={`M${s.w/2} 1 V34 M20 23 H${s.w-20}`} stroke="#8fbdc5" opacity=".55"/><rect x={s.w/2-11} y={s.h-20} width="22" height="10" rx="2" fill="#f4ae72" opacity=".7"/></g>}
 {active&&<ellipse cx={s.w/2} cy={s.h*.76} rx={s.w*.6} ry={s.h*.57} fill="none" stroke="#f9b374" strokeWidth="2" strokeDasharray="5 5"/>}
 {affected&&<circle cx={s.w*.88} cy="2" r="7" fill="#ef695d" className="crisis-pulse"/>}
 <rect x="-3" y={s.h+5} width={s.w+6} height="19" rx="3" fill="#101b27" fillOpacity=".94"/>
 <text x={s.w/2} y={s.h+18} fill={active?"#f6ac72":"#bed0d9"} fontSize="10" letterSpacing="1" fontFamily="monospace" textAnchor="middle">{m?.label || s.id.toUpperCase()}</text>
 </g>})}
 <g opacity=".35" fill="#df9473"><circle cx="60" cy="160" r="3"/><circle cx="630" cy="90" r="5"/><circle cx="645" cy="400" r="4"/><circle cx="95" cy="420" r="7"/></g>
 </svg>
 <div className="map-selected"><span className="selection-dot"/><div><strong>{current?.label || "SELECT A MODULE"}</strong><small>{current?`SYSTEM INTEGRITY ${current.health}%` : "Click a module to inspect"}</small></div><span className="map-tag">INSPECT</span></div>
 {crisis&&<div className="crisis-map-tag">⚠ ACTIVE INCIDENT</div>}
 </div>;
}
