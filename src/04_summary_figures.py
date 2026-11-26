"""
04_summary_figures.py — 결과 요약용 그림 2개 (outputs/public/, 병원명 없음)

  fig1_feature_comparison.png : 입점 vs 미입점 병원의 특성 비교 (특성마다 단위가 달라 작은 그림 4개로 나눔)
  fig2_score_bins.png         : 모델 점수 구간(5분위)별 실제 입점 비율 — 상위 구간일수록 높아지는지

점수는 교차검증 점수(학습에 쓰지 않은 병원으로 채점, 시드 10개 평균)를 쓴다.
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/public"

# 색 (참조 팔레트, 라이트 모드)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PARTNER, NONPARTNER = "#2a78d6", "#b9b7b0"      # 입점 = 파랑(강조), 미입점 = 회색(배경)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean(ax):
    """얇은 축, 연한 가로 격자만"""
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig1(df):
    panels = [("개원연차", "개원 연차 (평균, 년)", "{:.1f}년", 1),
              ("안과전문의수", "안과 전문의 수 (평균, 명)", "{:.1f}명", 1),
              ("경쟁밀도_1km", "반경 1km 안과 수 (평균, 곳)", "{:.1f}곳", 1),
              ("시력교정신고", "라식·라섹 시행 비율", "{:.0f}%", 100)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (col, title, fmt, scale) in zip(axes, panels):
        vals = [df.loc[df.y == 1, col].mean() * scale, df.loc[df.y == 0, col].mean() * scale]
        bars = ax.bar(["입점", "미입점"], vals, color=[PARTNER, NONPARTNER], width=0.55, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=11, color=INK, fontweight="bold")
        ax.set_title(title, fontsize=11.5, color=INK, pad=12, loc="left")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_yticklabels([])
        clean(ax)
        ax.yaxis.grid(False)          # 값은 막대 위에 직접 적었으니 격자 불필요
    fig.suptitle(f"입점 병원은 더 젊고, 크고, 밀집 상권에 있고, 라식·라섹을 한다  (서울 안과 의원 {len(df)}곳, 입점 {int(df.y.sum())}곳)",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.01, -0.04, "출처: 심평원 공개 자료, 병원 플랫폼 공개 목록(2026-11) · 개인 분석 · 공개 데이터 기반",
             fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_feature_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2(scores):
    s = scores.sort_values("검증점수", ascending=False).reset_index(drop=True)
    n = len(s)
    s["구간"] = pd.cut(np.arange(n), bins=5, labels=["상위 20%", "20~40%", "40~60%", "60~80%", "하위 20%"])
    g = s.groupby("구간", observed=True)["y"].agg(["mean", "sum", "count"])
    base = s["y"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(g.index.astype(str), g["mean"] * 100, color=PARTNER, width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, r["mean"] * 100 + 1.2, f"{r['mean']:.0%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, r["mean"] * 100 + 6.5, f"{int(r['sum'])}/{int(r['count'])}곳",
                ha="center", va="bottom", fontsize=9, color=INK2)
    ax.axhline(base * 100, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(4.42, base * 100 + 1.5, f"무작위 기준 {base:.1%}", ha="right", va="bottom", fontsize=9.5, color=INK2)
    ax.set_ylim(0, 85)
    ax.set_ylabel("실제 입점 비율 (%)")
    clean(ax)
    lift = g["mean"].iloc[0] / base
    ax.set_title(f"모델 점수가 높은 구간일수록 실제 입점 비율이 높다, 상위 20%는 무작위의 {lift:.1f}배",
                 fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.text(0.01, -0.03, "점수: 5-fold 교차검증 점수(시드 10개 평균), 서울 안과 의원 475곳 · 개인 분석 · 공개 데이터 기반",
             fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_score_bins.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return g, base


def variable_bins(df):
    """회귀 변수별로 구간을 나눠 구간마다 실제 입점 비율을 센다 (포트폴리오 페이지의 변수별 그래프용)"""
    specs = {
        "개원연차": ([-0.1, 5, 10, 15, 20, 30, 99], ["5년 이하", "5~10년", "10~15년", "15~20년", "20~30년", "30년 초과"]),
        "안과전문의수": ([0, 1, 2, 3, 5, 99], ["1명", "2명", "3명", "4~5명", "6명 이상"]),
        "경쟁밀도_1km": ([-1, 2, 5, 10, 20, 999], ["0~2곳", "3~5곳", "6~10곳", "11~20곳", "21곳 이상"]),
        "시력교정신고": ([-1, 0, 1], ["아니요", "예"]),
    }
    rows = []
    for col, (bins, labels) in specs.items():
        g = df.groupby(pd.cut(df[col], bins, labels=labels), observed=True)["y"].agg(["sum", "count"])
        r = np.corrcoef(df[col], df["y"])[0, 1]        # 변수와 입점 여부(0/1)의 상관계수
        for lab, row in g.iterrows():
            rows.append({"변수": col, "구간": lab, "입점": int(row["sum"]), "병원수": int(row["count"]),
                         "입점비율": round(row["sum"] / row["count"], 3), "상관계수": round(r, 2)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/variable_bins.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    model = load_module("model", "02_model.py")
    df = model.load_final()
    fig1(df)
    print(variable_bins(df).to_string(index=False))
    scores = pd.read_csv(ROOT / "data/processed/cycle3b_scores.csv")
    g, base = fig2(scores)
    print(g.round(3).to_string())
    print(f"무작위 기준 {base:.3f}")
