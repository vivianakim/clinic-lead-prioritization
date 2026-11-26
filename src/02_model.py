"""
02_model.py — 입점 가능성 모델 (가설 → 실행 → 검증 → 반영)

이 PC에서는 Windows 애플리케이션 제어 정책이 scipy의 DLL을 막아 scikit-learn·statsmodels를 쓸 수 없다.
그래서 같은 계산을 numpy로 직접 구현했다.
  - 로지스틱 회귀: 표준화 + L2 규제(C=1) + 클래스 가중치 'balanced' (scikit-learn 기본과 같은 목적함수)
  - 5-fold 층화 교차검증을 시드 10개로 반복
  - 계수 해석용 로지스틱 회귀(규제 없음): 오즈비, 95% 신뢰구간, p값 (statsmodels Logit과 같은 계산)

사용법:
  python -X utf8 src/02_model.py prep     # 준비: 입점 수, 그룹 비교표, VIF
  python -X utf8 src/02_model.py cycle1   # 사이클 1
  python -X utf8 src/02_model.py cycle2_review   # 사이클 1 반영 모델 + 틀린 병원 5+5곳
"""
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/eye_clinics.csv"
EXP_LOG = ROOT / "outputs/experiments.csv"
SEEDS = range(10)       # 교차검증 반복 시드
K = 5                   # 5-fold
TOP_SHARE = 0.20        # 상위 20%

LABELS = {"개원연차": "개원 연차(년)", "안과전문의수": "안과 전문의 수(명)",
          "수술실보유": "수술실 보유(1/0)", "경쟁밀도_1km": "반경 1km 안과 수(곳)",
          "홈페이지보유": "홈페이지 보유(1/0)"}


# ── 1. 데이터와 특성 변수 ────────────────────────────────────────
def load():
    df = pd.read_csv(DATA, dtype={"입점여부": str})
    df = df[df["입점여부"] != "보류"].copy()           # 보류 제외
    df["y"] = df["입점여부"].astype(int)
    df["경쟁밀도_1km"] = competition_density(df)
    return df.reset_index(drop=True)


# 시력교정·백내장 수술 레이저 모델명 (심평원 「의료장비 상세 현황」의 모델명으로 판별)
#   엑시머(라식·라섹): Allegretto/EX500, Amaris/Esiris/Keratom·Multiscan, MEL, VISX Star, Technolas/Teneo, EC-5000, LADARVision
#   펨토초(스마일·펨토라식·레이저백내장): VisuMax, Femto LDV, FS200, IntraLase/iFS, LenSx, LENSAR, Catalys, Victus, Elita
#   제외: YAG·SLT·망막 광응고 레이저(일반 진료용), 피부과 엑시머(XTRAC, EX-308)
SURGICAL_LASER = (r"ALLEGRETTO|EX500|AMARIS|ESIRIS|KERATOM|^MULTISCAN$|MEL ?[78]0|MFL ?80|VISX|VISIX|STAR S|^STAR$|"
                  r"TECHNOLAS|TENEO|EC-5000|LADARVIS|^EXIMER$|VISUMAX|FEMTO|FS200|INTRALASE|LENSX|LENSAR|CATALYS|VICTUS|ELITA")
SURGICAL_EXCLUDE = r"XTRAC|EX-308|ENDOSTAR"


def surgical_laser_flag(ykiho):
    """시력교정·백내장 수술 레이저를 1대 이상 보유하면 1"""
    path = next((ROOT / "data/raw/hira_equipment").glob("*.csv"))
    q = pd.read_csv(path, encoding="cp949", dtype=str,
                    usecols=["암호화된 요양기호", "장비대분류명", "모델명"])
    q = q[q["장비대분류명"].isin(["안과용 레이저수술기", "레이저수술기"])]
    model = q["모델명"].fillna("").str.upper().str.strip()
    hit = model.str.contains(SURGICAL_LASER, regex=True) & ~model.str.contains(SURGICAL_EXCLUDE, regex=True)
    ids = set(q.loc[hit, "암호화된 요양기호"])
    return ykiho.isin(ids).astype(int)


def npay_surgery_flags(df):
    """심평원 비급여 진료비용 공개(2026-11-25 화면 확인)에서 수술 항목을 신고한 병원 표시.
    결과에 암호화요양기호가 없어 '정규화한 병원명 + 구'로 매칭한다 (01_prepare.py의 norm_name 사용).
      시력교정신고: 라식·라섹 가격 신고 / 다초점백내장신고: 조절성(다초점) 인공수정체 가격 신고
      수술항목신고: 둘 중 하나라도 신고 (각막교차결합술 포함)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("prep", ROOT / "src/01_prepare.py")
    prep_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep_mod)
    path = next((ROOT / "data/raw/hira_npay").glob("hira_npay_seoul_eye_*.csv"))
    n = pd.read_csv(path, dtype=str)
    n["구"] = n["소재지"].str.split().str[1]
    n["key"] = n["의료기관명"].map(prep_mod.norm_name) + "|" + n["구"]
    keys = {
        "시력교정신고": set(n.loc[n["검색키워드"] == "시력교정술", "key"]),
        "다초점백내장신고": set(n.loc[n["소분류"] == "조절성 인공수정체", "key"]),
        "수술항목신고": set(n["key"]),
    }
    k = df["병원명"].map(prep_mod.norm_name) + "|" + df["구"]
    out = pd.DataFrame({c: k.isin(s).astype(int) for c, s in keys.items()}, index=df.index)
    # 매칭 점검용: 비급여 목록의 서울 '의원' 중 안과 리스트와 이어지지 않은 병원 수
    clinic_keys = set(n.loc[n["의료기관규모"] == "의원", "key"])
    out.attrs["unmatched_clinics"] = sorted(clinic_keys - set(k))
    return out


def competition_density(df, radius_km=1.0):
    """좌표(위도·경도)로 두 병원 사이 거리를 구해, 반경 1km 안에 있는 다른 안과 수를 센다."""
    lat = np.radians(df["위도"].to_numpy())
    lon = np.radians(df["경도"].to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371 * np.arcsin(np.sqrt(h))
    return (dist <= radius_km).sum(axis=1) - 1   # 자기 자신 제외


# ── 2. 로지스틱 회귀 (numpy 구현) ────────────────────────────────
def fit_logit(X, y, l2=0.0, sample_weight=None, iters=100):
    """뉴턴법으로 로지스틱 회귀를 푼다. X 첫 열은 절편(1). 절편에는 규제를 걸지 않는다."""
    n, p = X.shape
    w = np.ones(n) if sample_weight is None else sample_weight
    beta = np.zeros(p)
    pen = np.full(p, l2)
    pen[0] = 0.0
    for _ in range(iters):
        prob = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (w * (prob - y)) + pen * beta
        hess = (X * (w * prob * (1 - prob))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-9:
            break
    return beta, hess


def balanced_weights(y):
    """클래스 불균형 보정: 적은 쪽(입점) 병원에 더 큰 가중치 (n / (2 × 클래스 수))"""
    n, n1 = len(y), y.sum()
    return np.where(y == 1, n / (2 * n1), n / (2 * (n - n1)))


def predict_scores(X_tr, y_tr, X_te):
    """학습 데이터 기준으로 표준화 → L2(C=1)+balanced 로지스틱 회귀 → 검증 병원 점수"""
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1
    Z_tr = np.column_stack([np.ones(len(X_tr)), (X_tr - mu) / sd])
    Z_te = np.column_stack([np.ones(len(X_te)), (X_te - mu) / sd])
    beta, _ = fit_logit(Z_tr, y_tr, l2=1.0, sample_weight=balanced_weights(y_tr))
    return 1 / (1 + np.exp(-Z_te @ beta))


# ── 3. 평가 지표 ────────────────────────────────────────────────
def stratified_folds(y, k, seed):
    """입점/미입점 비율이 fold마다 같도록 나눈다."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def auc(y, s):
    """AUC = 입점 병원 하나와 미입점 병원 하나를 뽑았을 때 입점 쪽 점수가 더 높을 확률 (동점은 0.5)"""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):              # 동점 처리: 평균 순위
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def top_metrics(y, s):
    """점수 상위 20% 병원 안의 입점 비율(정밀도), 무작위 대비 배수, 전체 입점 중 포착률"""
    n_top = int(round(len(y) * TOP_SHARE))
    top = np.argsort(-s, kind="mergesort")[:n_top]
    precision = y[top].mean()
    base = y.mean()
    return {"precision": precision, "lift": precision / base, "capture": y[top].sum() / y.sum()}


def evaluate(df, features, cycle, hypothesis, note=""):
    """모든 사이클에서 똑같이 쓰는 평가 함수: 5-fold × 시드 10개, 검증 점수로만 채점"""
    X = df[features].to_numpy(float)
    y = df["y"].to_numpy()
    rows, oof_all = [], []
    for seed in SEEDS:
        fold = stratified_folds(y, K, seed)
        oof = np.zeros(len(y))
        for f in range(K):
            tr, te = fold != f, fold == f
            oof[te] = predict_scores(X[tr], y[tr], X[te])
        m = top_metrics(y, oof)
        m["auc"] = auc(y, oof)
        rows.append(m)
        oof_all.append(oof)
    r = pd.DataFrame(rows)
    summary = {
        "날짜": date.today().isoformat(), "사이클": cycle, "가설": hypothesis,
        "변수": "+".join(features), "병원수": len(y), "입점수": int(y.sum()),
        "기준_입점비율": round(y.mean(), 4),
        "상위20%_정밀도_평균": round(r.precision.mean(), 4),
        "상위20%_정밀도_최소": round(r.precision.min(), 4), "상위20%_정밀도_최대": round(r.precision.max(), 4),
        "배수_평균": round(r.lift.mean(), 3), "배수_최소": round(r.lift.min(), 3), "배수_최대": round(r.lift.max(), 3),
        "포착률_평균": round(r.capture.mean(), 4), "AUC_평균": round(r.auc.mean(), 4),
        "AUC_최소": round(r.auc.min(), 4), "AUC_최대": round(r.auc.max(), 4), "비고": note,
    }
    EXP_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(EXP_LOG, mode="a", index=False, header=not EXP_LOG.exists(),
                                   encoding="utf-8-sig")
    return summary, np.mean(oof_all, axis=0)


# ── 4. 계수 해석 (규제 없는 로지스틱 회귀, 원래 단위) ─────────────
def coef_table(df, features):
    X = np.column_stack([np.ones(len(df)), df[features].to_numpy(float)])
    y = df["y"].to_numpy()
    beta, hess = fit_logit(X, y)
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    z = beta / se
    p = [math.erfc(abs(v) / math.sqrt(2)) for v in z]
    t = pd.DataFrame({"변수": ["절편"] + features, "계수": beta, "표준오차": se, "z": z,
                      "오즈비": np.exp(beta),
                      "95%CI_하한": np.exp(beta - 1.96 * se), "95%CI_상한": np.exp(beta + 1.96 * se),
                      "p값": p})
    return t.iloc[1:].round(4)


def vif(df, features):
    """VIF: 한 변수를 나머지 변수로 얼마나 설명할 수 있는지 (5 이상이면 겹침이 큼)"""
    out = {}
    for f in features:
        others = [o for o in features if o != f]
        A = np.column_stack([np.ones(len(df)), df[others].to_numpy(float)])
        yv = df[f].to_numpy(float)
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        r2 = 1 - ((yv - A @ coef) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        out[f] = round(1 / (1 - r2), 2)
    return out


# ── 5. 실행 단계 ────────────────────────────────────────────────
CYCLE1 = ["개원연차", "경쟁밀도_1km", "안과전문의수", "수술실보유"]


def prep():
    df = load()
    raw = pd.read_csv(DATA, dtype={"입점여부": str})
    print("[입점여부 개수]", raw["입점여부"].value_counts().to_dict())
    print(f"분석 병원 {len(df)}곳, 입점 {df.y.sum()}곳 → 전체 입점 비율 {df.y.mean():.1%}")
    cols = CYCLE1 + ["홈페이지보유", "총의사수"]
    g = df.groupby("y")[cols].mean().T.round(2)
    g.columns = ["미입점 평균", "입점 평균"]
    g["차이(입점−미입점)"] = (g["입점 평균"] - g["미입점 평균"]).round(2)
    print("\n[입점/미입점 그룹별 평균]")
    print(g.to_string())
    print("\n[중앙값] 개원연차·경쟁밀도:", df.groupby("y")[["개원연차", "경쟁밀도_1km"]].median().to_dict())
    print("\n[VIF] (5 이상이면 변수끼리 많이 겹침)", vif(df, CYCLE1))
    print("\n[상관계수]")
    print(df[CYCLE1].corr().round(2).to_string())


def cycle1():
    df = load()
    hyp = "개원 초기·경쟁 심함·규모 큼·수술실 있음 → 입점"
    s, oof = evaluate(df, CYCLE1, "1", hyp)
    print_summary(s)
    print("\n[계수 해석] 오즈비 = 변수가 1 늘 때 입점 가능성(오즈)이 몇 배가 되는지")
    print(coef_table(df, CYCLE1).to_string(index=False))

    # 참고: 개원 1년 미만(개원연차 0) 병원 제외 — 아직 플랫폼에 등록되지 않았을 수 있음
    sub = df[df["개원연차"] >= 1].reset_index(drop=True)
    s2, _ = evaluate(sub, CYCLE1, "1-참고", hyp, note="개원연차 0(2026년 개원) 제외")
    print("\n[참고] 2026년 개원 병원 제외 시")
    print_summary(s2)
    print(coef_table(sub, CYCLE1).to_string(index=False))

    # 사이클 2에서 볼 병원을 위해 검증 점수 저장 (병원명 포함 → data/processed)
    df.assign(검증점수=oof).to_csv(ROOT / "data/processed/cycle1_scores.csv", index=False, encoding="utf-8-sig")


CYCLE1_FINAL = ["개원연차", "경쟁밀도_1km", "안과전문의수"]   # 사이클 1 반영: 수술실 보유 제외


def cycle2_review():
    """사이클 1 반영 모델로 다시 채점하고, 모델이 틀린 병원 5+5곳을 뽑는다 (직접 살펴볼 목록)"""
    df = load()
    s, oof = evaluate(df, CYCLE1_FINAL, "1-반영", "수술실 보유 제외 (효과 불확실, 빼도 배수 차이 없음)")
    print_summary(s)
    print(coef_table(df, CYCLE1_FINAL).to_string(index=False))

    df["검증점수"] = oof
    df["점수순위"] = df["검증점수"].rank(ascending=False, method="first").astype(int)
    cols = ["점수순위", "검증점수", "병원명", "플랫폼병원명", "구", "주소", "개원연차", "안과전문의수",
            "경쟁밀도_1km", "수술실보유", "홈페이지보유"]
    fp = df[df["y"] == 0].nsmallest(5, "점수순위")[cols]   # 점수 높은데 미입점
    fn = df[df["y"] == 1].nlargest(5, "점수순위")[cols]    # 점수 낮은데 입점
    df.to_csv(ROOT / "data/processed/cycle1_scores.csv", index=False, encoding="utf-8-sig")
    pd.concat([fp.assign(구분="점수 높은데 미입점"), fn.assign(구분="점수 낮은데 입점")]).to_csv(
        ROOT / "data/processed/cycle2_review.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"\n[점수 높은데 미입점 5곳] (상위 20% = {int(round(len(df) * TOP_SHARE))}위까지)")
    print(fp.round(3).to_string(index=False))
    print("\n[점수 낮은데 입점 5곳]")
    print(fn.round(3).to_string(index=False))


CYCLE2 = CYCLE1_FINAL + ["홈페이지보유"]

# ── 사이클 2 확정 (2026-11-25) ──
# 대상: 의원급만 (대형병원=종별 '병원' 제외) → 채택
# 홈페이지 보유 → 기각 (배수 3.48 → 3.39, 오르지 않음). 판단 근거는 outputs/experiment_notes.md
# ── 사이클 3 확정 (2026-11-25) ──
# 시력교정(라식·라섹) 비급여 가격 신고 여부 → 채택 (오즈비 7.2, AUC 0.886 → 0.903)
FINAL_FEATURES_C2 = CYCLE1_FINAL
FINAL_FEATURES = CYCLE1_FINAL + ["시력교정신고"]


def load_final():
    """확정된 분석 대상: 보류 제외 + 의원급만 (+ 비급여 신고 변수)"""
    df = load()
    df = df[df["종별"] == "의원"].reset_index(drop=True)
    return df.join(npay_surgery_flags(df))


def cycle2():
    """사이클 2: 대형병원(종별=병원) 제외 + 홈페이지 보유 추가.
    두 변화의 효과를 나누어 보려고, 같은 병원(의원급)에서 기존 3변수 모델도 함께 평가한다."""
    df = load()
    print("[제외한 병원급 안과]", df.loc[df["종별"] != "의원", ["병원명", "y"]].values.tolist())
    df = df[df["종별"] == "의원"].reset_index(drop=True)

    s0, _ = evaluate(df, CYCLE1_FINAL, "2-기준", "대형병원(종별=병원) 제외만 적용", note="의원급만, 3변수")
    print("\n[2-기준] 의원급만, 기존 3변수")
    print_summary(s0)

    hyp = "대형병원은 플랫폼이 필요 없음 + 홈페이지가 있는(온라인 마케팅에 적극적인) 병원이 입점"
    s1, _ = evaluate(df, CYCLE2, "2", hyp, note="의원급만, 3변수+홈페이지보유")
    print("\n[2] 의원급만, 3변수 + 홈페이지 보유")
    print_summary(s1)
    print(coef_table(df, CYCLE2).to_string(index=False))
    print("\n[VIF]", vif(df, CYCLE2))
    print("\n[홈페이지 보유별 입점 비율]")
    print(df.groupby("홈페이지보유")["y"].agg(병원수="count", 입점수="sum", 입점비율="mean").round(3).to_string())


CYCLE3 = FINAL_FEATURES_C2 + ["수술레이저"]


def cycle3():
    """사이클 3: 수술 주력 병원이 입점한다 — 시력교정·백내장 수술 레이저 보유로 측정"""
    df = load_final()
    df["수술레이저"] = surgical_laser_flag(df["암호화요양기호"])
    print("[수술레이저 보유별 입점 비율]")
    print(df.groupby("수술레이저")["y"].agg(병원수="count", 입점수="sum", 입점비율="mean").round(3).to_string())
    df["규모"] = pd.cut(df["안과전문의수"], [0, 1, 3, 100], labels=["1명", "2~3명", "4명+"])
    print("\n[전문의 규모별]")
    print(df.groupby(["규모", "수술레이저"], observed=True)["y"].agg(병원수="count", 입점비율="mean").round(2).to_string())

    hyp = "수술 주력 병원(시력교정·백내장 수술 레이저 보유)이 입점"
    s, oof = evaluate(df, CYCLE3, "3", hyp, note="의원급 475곳, 확정 3변수+수술레이저 (심평원 의료장비 2025-12-31)")
    print("\n[3] 확정 3변수 + 수술레이저  (비교: 2-기준 3.48배, 정밀도 61.6%, AUC 0.886)")
    print_summary(s)
    print(coef_table(df, CYCLE3).to_string(index=False))
    print("\n[VIF]", vif(df, CYCLE3))
    df.assign(검증점수=oof).to_csv(ROOT / "data/processed/cycle3_scores.csv", index=False, encoding="utf-8-sig")


def cycle3b():
    """사이클 3-b: 수술 주력을 '심평원 비급여 가격 신고 여부'로 측정 (장비보다 직접적인 근거)"""
    df = load_final()
    df["수술레이저"] = surgical_laser_flag(df["암호화요양기호"])
    df["규모"] = pd.cut(df["안과전문의수"], [0, 1, 3, 100], labels=["1명", "2~3명", "4명+"])
    print("[전문의 규모별 — 시력교정(라식·라섹) 가격 신고 여부]")
    print(df.groupby(["규모", "시력교정신고"], observed=True)["y"].agg(병원수="count", 입점수="sum", 입점비율="mean")
          .round(2).to_string())
    print("\n(비교: 2-기준 3.48배 / 3(수술레이저) 3.54배)")
    variants = [
        ("3b-시력교정", FINAL_FEATURES_C2 + ["시력교정신고"], "라식·라섹 비급여 가격을 신고한(시력교정 수술하는) 병원이 입점"),
        ("3b-다초점", FINAL_FEATURES_C2 + ["다초점백내장신고"], "다초점 백내장 렌즈 가격을 신고한 병원이 입점"),
        ("3b-둘다", FINAL_FEATURES_C2 + ["시력교정신고", "다초점백내장신고"], "시력교정·다초점 백내장 수술 병원이 입점"),
    ]
    for name, feats, hyp in variants:
        s, oof = evaluate(df, feats, name, hyp, note="의원급 475곳, 심평원 비급여 공개 화면 2026-11-25")
        print(f"\n[{name}] {' + '.join(feats)}")
        print_summary(s)
        print(coef_table(df, feats).to_string(index=False))
        print("VIF", vif(df, feats))
        if name == "3b-시력교정":
            df.assign(검증점수=oof).to_csv(ROOT / "data/processed/cycle3b_scores.csv", index=False,
                                          encoding="utf-8-sig")


def reason_line(r, contrib):
    """점수를 가장 많이 올린 특성 2개를 골라 한 줄 이유로 만든다 (기여도 = 표준화 계수 × 표준화 값)"""
    text = {
        "시력교정신고": "라식·라섹 시행(비급여 가격 신고)",
        "경쟁밀도_1km": f"반경 1km 안과 {int(r['경쟁밀도_1km'])}곳 밀집",
        "안과전문의수": f"안과 전문의 {int(r['안과전문의수'])}명",
        "개원연차": f"개원 {int(r['개원연차'])}년차",
    }
    top = [f for f, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][:2]
    return " · ".join(text[f] for f in top) if top else "뚜렷한 강점 없음"


def recommend_product(r):
    """추천 상품 (규칙 기반, 가정): 시력교정 시행 → 시술 할인 이벤트 광고 / 다초점 백내장만 → 상담·견적 연결 / 그 외 → 검진·진료 할인 상품"""
    if r["시력교정신고"] == 1:
        return "시술 할인 이벤트 광고 (시력교정 이벤트 노출)"
    if r["다초점백내장신고"] == 1:
        return "상담·견적 연결 (백내장·노안 상담)"
    return "검진·진료 할인 상품"


# ── 공개본용 구간화: 구 + 정확한 수치를 함께 공개하면 공공데이터로 병원을 거꾸로 찾을 수 있어 구간으로만 보여 준다
def bucket_years(v):
    return "5년 이하" if v <= 5 else "6~10년" if v <= 10 else "11~20년" if v <= 20 else "20년 초과"


def bucket_doctors(v):
    return "1명" if v <= 1 else "2~3명" if v <= 3 else "4명 이상"


def bucket_density(v):
    return "5곳 이하" if v <= 5 else "6~10곳" if v <= 10 else "11곳 이상"


def public_view(out):
    """내부용 Top 30 → 공개용: 병원명·점수를 빼고, 수치와 '점수가 높은 이유' 속 숫자를 구간으로 바꾼다"""
    pub = out.drop(columns=["병원명", "점수"]).copy()
    pub["개원연차"] = pub["개원연차"].map(bucket_years)
    pub["의사수(안과전문의)"] = pub["의사수(안과전문의)"].map(bucket_doctors)
    pub["경쟁밀도(1km 안과 수)"] = pub["경쟁밀도(1km 안과 수)"].map(bucket_density)
    pub["점수가 높은 이유"] = (pub["점수가 높은 이유"]
                         .str.replace(r"안과 전문의 (\d+)명", lambda m: "안과 전문의 " + bucket_doctors(int(m.group(1))), regex=True)
                         .str.replace(r"반경 1km 안과 (\d+)곳 밀집", lambda m: "반경 1km 안과 " + bucket_density(int(m.group(1))), regex=True)
                         .str.replace(r"개원 (\d+)년차", lambda m: "개원 " + bucket_years(int(m.group(1))), regex=True))
    return pub


def final():
    """확정 모델을 전체 475곳으로 다시 학습 → 미입점 병원 점수 상위 30곳"""
    df = load_final()
    X = df[FINAL_FEATURES].to_numpy(float)
    y = df["y"].to_numpy()
    # 전체 데이터로 학습 (교차검증 지표는 3b-시력교정 결과를 그대로 보고)
    mu, sd = X.mean(0), X.std(0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, _ = fit_logit(Z, y, l2=1.0, sample_weight=balanced_weights(y))
    df["점수"] = 1 / (1 + np.exp(-Z @ beta))
    contrib = (Z[:, 1:] * beta[1:])               # 병원별 특성 기여도

    cand = df[df["y"] == 0].copy()
    cand["이유"] = [reason_line(df.loc[i], dict(zip(FINAL_FEATURES, contrib[i]))) for i in cand.index]
    cand["추천 상품"] = cand.apply(recommend_product, axis=1)
    top = cand.nlargest(30, "점수").reset_index(drop=True)
    top.insert(0, "순위", range(1, len(top) + 1))
    # 익명명: 구별로 점수 순서대로 A, B, C… (예: 강남구 A안과)
    top["익명명"] = top.groupby("구").cumcount().map(lambda i: chr(65 + i)) .radd(top["구"] + " ") + "안과"
    out = top[["순위", "병원명", "익명명", "구", "개원연차", "안과전문의수", "경쟁밀도_1km", "시력교정신고", "점수", "이유", "추천 상품"]]
    out = out.rename(columns={"안과전문의수": "의사수(안과전문의)", "경쟁밀도_1km": "경쟁밀도(1km 안과 수)",
                              "시력교정신고": "라식·라섹 시행", "이유": "점수가 높은 이유", "추천 상품": "추천 상품(가정)"})
    out["점수"] = out["점수"].round(3)
    out.to_csv(ROOT / "outputs/top30.csv", index=False, encoding="utf-8-sig")
    public_view(out).to_csv(ROOT / "outputs/public/top30_public.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(out.drop(columns=["병원명"]).to_string(index=False))
    print("\n[최종 모델 계수]")
    coefs = coef_table(df, FINAL_FEATURES)
    # 회귀분석 결과표: 계수·표준오차·z·p값·오즈비 + 변수 간 겹침(VIF) + 입점/미입점 평균
    coefs["VIF"] = coefs["변수"].map(vif(df, FINAL_FEATURES)).astype(float)
    coefs["입점_평균"] = coefs["변수"].map(df[df["y"] == 1][FINAL_FEATURES].mean()).round(2)
    coefs["미입점_평균"] = coefs["변수"].map(df[df["y"] == 0][FINAL_FEATURES].mean()).round(2)
    coefs.to_csv(ROOT / "outputs/regression_table.csv", index=False, encoding="utf-8-sig")
    print(coefs.to_string(index=False))
    print("\nTop 30 구 분포:", top["구"].value_counts().to_dict())
    print("Top 30 추천 상품:", out["추천 상품(가정)"].value_counts().to_dict())


def print_summary(s):
    print(f"병원 {s['병원수']}곳, 입점 {s['입점수']}곳 (무작위 기준 {s['기준_입점비율']:.1%})")
    print(f"상위 20% 정밀도 {s['상위20%_정밀도_평균']:.1%} (범위 {s['상위20%_정밀도_최소']:.1%}~{s['상위20%_정밀도_최대']:.1%})")
    print(f"무작위 대비 배수 {s['배수_평균']:.2f}배 (범위 {s['배수_최소']:.2f}~{s['배수_최대']:.2f})")
    print(f"상위 20% 포착률 {s['포착률_평균']:.1%}, AUC {s['AUC_평균']:.3f} (범위 {s['AUC_최소']:.3f}~{s['AUC_최대']:.3f})")


if __name__ == "__main__":
    {"prep": prep, "cycle1": cycle1, "cycle2_review": cycle2_review, "cycle2": cycle2, "cycle3": cycle3, "cycle3b": cycle3b, "final": final}[sys.argv[1]]()
