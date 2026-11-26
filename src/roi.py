"""
roi.py — 손익분기 예약 건수 계산기

  손익분기 예약 건수 = 월 광고비 ÷ (시술가 × 공헌이익률)

- 모든 기본값은 가정이다. 플랫폼의 실제 광고 단가는 알 수 없다.
- 환자 1명당 수수료 구조는 의료법상 환자 알선 논란이 있어 쓰지 않고, 정액 광고 기준으로만 계산한다.
- 실제 값을 넣으면 바로 다시 계산된다:  python -X utf8 src/roi.py --ad 1500000 --price 900000 --margin 0.45
"""
import argparse
import math

# ── 기본 가정값 (가정) ─────────────────────────────────────────
DEFAULTS = {
    "ad_cost": 1_000_000,   # 월 광고비 100만원 (가정, 실제 단가는 알 수 없음)
    "price": 700_000,       # 시술가: 라섹 70만원 (가정, 공개된 시력교정 이벤트 가격대 참고)
    "margin": 0.50,         # 공헌이익률 50% (가정)
}
SENSITIVITY_MARGINS = [0.30, 0.40, 0.50, 0.60]   # 공헌이익률 민감도 30~60% (가정)


def break_even(ad_cost=DEFAULTS["ad_cost"], price=DEFAULTS["price"], margin=DEFAULTS["margin"]):
    """손익분기 예약 건수. (소수, 올림한 정수) 둘 다 돌려준다."""
    contribution = price * margin            # 예약 1건이 남기는 공헌이익
    raw = ad_cost / contribution
    return raw, math.ceil(raw)


def won(x):
    """원 단위를 '100만원'처럼 읽기 쉽게"""
    return f"{x / 10_000:,.0f}만원"


def sensitivity_rows(ad_cost=DEFAULTS["ad_cost"], price=DEFAULTS["price"], margins=SENSITIVITY_MARGINS):
    rows = []
    for m in margins:
        raw, n = break_even(ad_cost, price, m)
        rows.append({"공헌이익률": m, "건당 공헌이익": price * m, "손익분기(소수)": raw, "손익분기 예약 건수": n})
    return rows


def roi_markdown(ad_cost=DEFAULTS["ad_cost"], price=DEFAULTS["price"], margin=DEFAULTS["margin"],
                 procedure="라섹"):
    """제안서에 넣을 ROI 표 (마크다운). 가정값에는 모두 '(가정)' 표시."""
    raw, n = break_even(ad_cost, price, margin)
    lines = [
        f"| 항목 | 값 |",
        f"| --- | --- |",
        f"| 월 광고비 | {won(ad_cost)} (가정) |",
        f"| 시술가 ({procedure}) | {won(price)} (가정) |",
        f"| 공헌이익률 | {margin:.0%} (가정) |",
        f"| 예약 1건 공헌이익 | {won(price * margin)} (가정) |",
        f"| **손익분기 예약 건수** | **월 {n}건** (계산값 {raw:.1f}건, 가정 기준) |",
        "",
        f"공헌이익률에 따른 손익분기 (월 광고비 {won(ad_cost)}, {procedure} {won(price)} 기준, 모두 가정)",
        "",
        "| 공헌이익률 (가정) | 30% | 40% | 50% | 60% |",
        "| --- | --- | --- | --- | --- |",
        "| 손익분기 예약 건수 | " + " | ".join(f"월 {r['손익분기 예약 건수']}건"
                                         for r in sensitivity_rows(ad_cost, price)) + " |",
    ]
    return "\n".join(lines), n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="손익분기 예약 건수 계산 (기본값은 모두 가정)")
    ap.add_argument("--ad", type=int, default=DEFAULTS["ad_cost"], help="월 광고비(원)")
    ap.add_argument("--price", type=int, default=DEFAULTS["price"], help="시술가(원)")
    ap.add_argument("--margin", type=float, default=DEFAULTS["margin"], help="공헌이익률(0~1)")
    a = ap.parse_args()
    md, _ = roi_markdown(a.ad, a.price, a.margin)
    print(md)
