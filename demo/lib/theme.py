"""lib/theme.py : identité visuelle partagée (vive, arrondie, sans emoji)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

INK        = "#181A20"
INK_SOFT   = "#5B5F6B"
INK_FAINT  = "#8B8F9B"
LINE       = "#E3E3EC"
LINE_SOFT  = "#EEEEF5"
BG         = "#F5F6FA"
PANEL      = "#FFFFFF"

ACCENT     = "#6D28D9"
ACCENT_SOFT= "#EDE7FB"

MLP        = "#2563EB"
LOGREG     = "#059669"
XGB        = "#F59E0B"
DANGER     = "#DC2626"
DANGER_SOFT= "#FBE7E6"
SAFE       = "#0891B2"
SAFE_SOFT  = "#E1F2F5"
PENDING    = "#9A9AA5"
PENDING_SOFT = "#EEEEF3"

MODEL_COLOR = {"MLP": MLP, "LogReg": LOGREG, "XGBoost": XGB}
MONO = '"SF Mono","Cascadia Code","Consolas",monospace'


def inject_css():
    st.markdown(f"""
    <style>
    .block-container {{ padding-top: 1.6rem; max-width: 1180px; }}
    h1, h2, h3 {{ letter-spacing:-.01em; }}
    .banner {{ font-size:30px; font-weight:800; margin:0 0 2px; color:{INK}; }}
    .subtitle {{ font-size:13.5px; color:{INK_FAINT}; margin-bottom:18px; }}
    .card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:20px 22px;
             box-shadow:0 1px 2px rgba(20,20,40,.03); }}
    .stat-tile {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:14px 16px; }}
    .stat-tile .label {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:{INK_FAINT}; font-weight:700; }}
    .stat-tile .value {{ font-size:23px; font-weight:800; margin-top:4px; color:{INK}; }}
    .badge {{ display:inline-flex; align-items:center; gap:5px; font-size:10.5px; font-weight:700;
              letter-spacing:.03em; padding:4px 10px; border-radius:999px; text-transform:uppercase; }}
    .badge-final {{ background:{SAFE_SOFT}; color:{SAFE}; }}
    .badge-prov  {{ background:#FEF3C7; color:#92650B; }}
    .badge-pending {{ background:{PENDING_SOFT}; color:{PENDING}; }}
    .badge-danger {{ background:{DANGER_SOFT}; color:{DANGER}; }}
    .margin-axis {{ position:relative; height:36px; margin:12px 0 6px; background:{LINE_SOFT};
                     border-radius:12px; overflow:hidden; }}
    .margin-axis .zone-evade {{ position:absolute; left:0; top:0; bottom:0; width:50%; background:{DANGER_SOFT}; }}
    .margin-axis .zone-det   {{ position:absolute; right:0; top:0; bottom:0; width:50%; background:{SAFE_SOFT}; }}
    .margin-axis .zero {{ position:absolute; left:50%; top:0; bottom:0; width:2px; background:{INK_FAINT}; opacity:.4; }}
    .margin-axis .marker {{ position:absolute; top:5px; width:14px; height:14px; border-radius:50%;
                             border:3px solid {PANEL}; box-shadow:0 2px 6px rgba(20,20,40,.25);
                             transition:left .5s cubic-bezier(.2,.8,.25,1); }}
    .stage {{ border:1px solid {LINE}; border-radius:12px; padding:10px 8px; background:{ACCENT_SOFT};
               text-align:center; font-size:11px; }}
    .stage.active {{ border-color:{DANGER}; background:{DANGER_SOFT}; }}
    .stage .sid {{ font-family:{MONO}; font-size:10.5px; color:{ACCENT}; font-weight:800; }}
    .result-panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:20px; padding:24px 26px; }}
    hr.thin {{ border:none; border-top:1px solid {LINE}; margin:22px 0; }}
    div[data-testid="stButton"] > button {{ border-radius:10px; }}
    </style>
    """, unsafe_allow_html=True)


def banner(title: str, subtitle: str = ""):
    st.markdown(f'<div class="banner">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<p style="font-size:11.5px;color:{INK_FAINT};font-style:italic;margin-top:6px;">{text}</p>',
                unsafe_allow_html=True)


def badge(text: str, kind: str = "pending") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def stat_tile(label: str, value: str, sub: str = ""):
    st.markdown(f'''
    <div class="stat-tile">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {f'<div style="font-size:11.5px;color:{INK_SOFT};margin-top:3px;">{sub}</div>' if sub else ""}
    </div>''', unsafe_allow_html=True)


def asr_curve_chart(success: pd.Series, current_idx: int, color: str):
    """Courbe réelle d'ASR cumulé sur les échantillons vus, avec un point qui se
    déplace sur la courbe au fil du curseur d'échantillon."""
    curve = pd.DataFrame({
        "echantillon": range(1, len(success) + 1),
        "asr": success.astype(float).expanding().mean().values * 100,
    })
    line = alt.Chart(curve).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X("echantillon:Q", title="Échantillons vus", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("asr:Q", title="ASR cumulé (%)", scale=alt.Scale(domain=[0, 100])),
    )
    point = alt.Chart(curve.iloc[[current_idx]]).mark_point(
        color=DANGER, size=160, filled=True,
    ).encode(x="echantillon:Q", y="asr:Q")
    st.altair_chart((line + point).properties(height=220), use_container_width=True)


def asr_bar_chart(values: dict, colors: dict | None = None):
    colors = colors or MODEL_COLOR
    df = pd.DataFrame([{"modele": k, "asr": v} for k, v in values.items() if v is not None])
    if df.empty:
        return
    chart = alt.Chart(df).mark_bar(size=34, cornerRadiusEnd=8).encode(
        x=alt.X("modele:N", title=None),
        y=alt.Y("asr:Q", title="ASR (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("modele:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())),
                         legend=None),
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)


def margin_axis(margin: float, color: str) -> str:
    pct = min(96, max(4, 50 + (margin / 25) * 46))
    return f'''
    <div class="margin-axis">
      <div class="zone-evade"></div><div class="zone-det"></div><div class="zero"></div>
      <div class="marker" style="left:calc({pct}% - 7px);background:{color}"></div>
    </div>'''
