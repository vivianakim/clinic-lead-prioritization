"""
01_prepare.py — 서울 안과 리스트 만들기 + 플랫폼 목록과 매칭

[입력]
- data/raw/hira_hospitals/ : 심평원 기본정보(1), 시설정보(3), 진료과목정보(5)
- data/raw/platform_listing.csv : 00_collect_partners.py 결과
- data/processed/manual_matches.csv (있으면) : 사람이 직접 확인한 매칭

[출력]
- data/processed/eye_clinics.csv : 분석용 서울 안과 리스트
- data/processed/review_*.csv   : 직접 확인이 필요한 목록
"""
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hira_hospitals"
PROC = ROOT / "data/processed"
BASE_YEAR = 2026          # 개원연차 = 2026 − 개설연도
CHECKED_DATE = "2026-11-24"


def read_hira(prefix):
    """심평원 파일은 CP949 인코딩. 파일명 앞 번호로 찾는다."""
    path = next(RAW.glob(f"{prefix}.*.csv"))
    return pd.read_csv(path, encoding="cp949", dtype=str)


# ── 1. 서울 안과 리스트 ─────────────────────────────────────────
def build_seoul_eye():
    base = read_hira("1")
    subj = read_hira("5")
    fac = read_hira("3")

    # 안과 전문의 수: 진료과목정보에서 진료과목코드명 = 안과
    subj["과목별 전문의수"] = pd.to_numeric(subj["과목별 전문의수"])
    eye = (subj[subj["진료과목코드명"] == "안과"]
           [["암호화요양기호", "과목별 전문의수"]]
           .rename(columns={"과목별 전문의수": "안과전문의수"}))

    # 판별 기준: 서울 + 안과 전문의 1명 이상 + 이름에 '안과' (의원·병원 구분 없음)
    df = base.merge(eye, on="암호화요양기호")
    df = df[(df["시도코드명"] == "서울")
            & (df["안과전문의수"] >= 1)
            & df["요양기관명"].str.contains("안과")].copy()

    # 수술실 보유: 시설정보의 수술실병상수 > 0
    fac["수술실병상수"] = pd.to_numeric(fac["수술실병상수"])
    df = df.merge(fac[["암호화요양기호", "수술실병상수"]], on="암호화요양기호", how="left")

    # 파생 변수
    df["개원연차"] = BASE_YEAR - pd.to_datetime(df["개설일자"]).dt.year
    df["총의사수"] = pd.to_numeric(df["총의사수"])
    df["수술실보유"] = (df["수술실병상수"].fillna(0) > 0).astype(int)
    df["홈페이지보유"] = df["병원홈페이지"].notna().astype(int)
    df["경도"] = pd.to_numeric(df["좌표(X)"])   # X = 경도 (126.8~127.2)
    df["위도"] = pd.to_numeric(df["좌표(Y)"])   # Y = 위도 (37.4~37.7)
    df = df.rename(columns={"요양기관명": "병원명", "시군구코드명": "구", "읍면동": "동", "종별코드명": "종별"})
    return df


# ── 2. 병원명 정규화 ────────────────────────────────────────────
def norm_name(s):
    """공백·괄호(내용 포함)·특수문자 제거, '안과의원/안과병원/의원' → 통일, 영문 소문자"""
    # '씨케이(CK)성모' → 'CK성모': 한글 표기 뒤 괄호 속 영문이 있으면 영문을 남김 (플랫폼은 영문 표기를 씀)
    s = re.sub(r"[가-힣]+\(([A-Za-z0-9]+)\)", r"\1", str(s))
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[\s·.\-&]", "", s).lower()
    s = s.replace("안과의원", "안과").replace("안과병원", "안과")
    s = re.sub(r"의원$", "", s)
    return s


def base_name(s):
    """플랫폼 지점명 떼기: '○○안과의원 반포점' → '○○안과의원' (첫 단어가 의원·병원·안과로 끝날 때만)"""
    first = str(s).split(" ")[0]
    if " " in str(s) and first.endswith(("의원", "병원", "안과")):
        return first
    return str(s)


def norm_dong(s):
    """행정동(역삼1동)과 법정동(역삼동)을 비교할 수 있게 숫자·'본'·점을 제거"""
    s = re.sub(r"[\d.·]", "", str(s))
    return s.replace("본동", "동").replace("가동", "동")


def similarity(a, b):
    """두 정규화 이름의 유사도(0~1). 한쪽이 다른 쪽을 포함하면 지점명 차이로 보고 가산."""
    r = SequenceMatcher(None, a, b).ratio()
    if a and b and (a in b or b in a):
        r = max(r, 0.9)
    return r


# ── 3. 매칭 ─────────────────────────────────────────────────────
def match(hira, plat):
    hira = hira.copy()
    plat = plat.copy()
    hira["key"] = hira["병원명"].map(norm_name)
    plat["key"] = plat["병원명"].map(norm_name)
    plat["base_key"] = plat["병원명"].map(lambda s: norm_name(base_name(s)))
    hira["dong_k"] = hira["동"].map(norm_dong)
    plat["dong_k"] = plat["동"].map(norm_dong)

    hira["플랫폼id"] = pd.NA
    hira["매칭방식"] = pd.NA
    used = set()

    # (0) 사람이 직접 확인한 매칭이 있으면 가장 먼저 적용
    manual_path = PROC / "manual_matches.csv"
    if manual_path.exists():
        man = pd.read_csv(manual_path, dtype=str)
        for _, m in man.iterrows():
            idx = hira.index[hira["암호화요양기호"] == m["암호화요양기호"]]
            if len(idx) == 0:
                continue
            if pd.isna(m.get("플랫폼id")) or str(m["플랫폼id"]).strip() in ("", "없음"):
                hira.loc[idx, "매칭방식"] = "수동(목록없음)"
            else:
                hira.loc[idx, ["플랫폼id", "매칭방식"]] = [int(m["플랫폼id"]), "수동"]
                used.add(int(m["플랫폼id"]))

    # (1) 완전일치: 정규화 이름 + 구가 같고, 후보가 1곳뿐일 때
    for i, h in hira[hira["매칭방식"].isna()].iterrows():
        c = plat[(plat["key"] == h["key"]) & (plat["구"] == h["구"]) & ~plat["id"].isin(used)]
        if len(c) > 1:  # 같은 구에 같은 이름이 여럿이면 동까지 비교
            c = c[c["dong_k"] == h["dong_k"]]
        if len(c) == 1:
            hira.loc[i, ["플랫폼id", "매칭방식"]] = [int(c["id"].iloc[0]), "완전일치"]
            used.add(int(c["id"].iloc[0]))

    # (1b) 지점명만 다른 경우: 플랫폼 이름에서 지점명을 떼면 같은 구 안에서 정확히 1곳과 같을 때
    for i, h in hira[hira["매칭방식"].isna()].iterrows():
        c = plat[(plat["base_key"] == h["key"]) & (plat["구"] == h["구"]) & ~plat["id"].isin(used)]
        if len(c) == 1:
            hira.loc[i, ["플랫폼id", "매칭방식"]] = [int(c["id"].iloc[0]), "완전일치(지점명)"]
            used.add(int(c["id"].iloc[0]))

    # (2) 유사: 같은 구 안에서 유사도 최고 후보. 0.85 이상 + 동 일치 → 유사, 그 외 0.7 이상 → 보류(직접 확인)
    review = []
    for i, h in hira[hira["매칭방식"].isna()].iterrows():
        c = plat[(plat["구"] == h["구"]) & ~plat["id"].isin(used)].copy()
        if c.empty:
            continue
        c["sim"] = c["key"].map(lambda k: similarity(h["key"], k))
        best = c.sort_values("sim", ascending=False).iloc[0]
        same_dong = best["dong_k"] == h["dong_k"]
        if best["sim"] >= 0.85 and same_dong:
            hira.loc[i, ["플랫폼id", "매칭방식"]] = [int(best["id"]), "유사"]
            used.add(int(best["id"]))
        elif best["sim"] >= 0.7:
            hira.loc[i, "매칭방식"] = "보류"
        if best["sim"] >= 0.7 and not (best["sim"] >= 0.85 and same_dong):
            review.append({"암호화요양기호": h["암호화요양기호"], "심평원_병원명": h["병원명"], "구": h["구"],
                           "심평원_동": h["동"], "후보_플랫폼id": int(best["id"]), "후보_병원명": best["병원명"],
                           "후보_동": best["동"], "후보_제휴": int(best["is_partner"]),
                           "유사도": round(best["sim"], 2)})

    # (2b) 구가 다른데 이름이 같은 경우 (이전·주소 표기 차이 가능) → 자동 매칭하지 않고 보류 + 확인 목록
    for i, h in hira[hira["매칭방식"].isna()].iterrows():
        c = plat[((plat["key"] == h["key"]) | (plat["base_key"] == h["key"])) & ~plat["id"].isin(used)]
        if len(c) == 1:
            best = c.iloc[0]
            hira.loc[i, "매칭방식"] = "보류"
            review.append({"암호화요양기호": h["암호화요양기호"], "심평원_병원명": h["병원명"], "구": h["구"],
                           "심평원_동": h["동"], "후보_플랫폼id": int(best["id"]), "후보_병원명": best["병원명"],
                           "후보_동": f'{best["구"]} {best["동"]} (구 다름)', "후보_제휴": int(best["is_partner"]),
                           "유사도": 1.0})

    # (3) 남은 병원 = 플랫폼 목록에서 찾지 못함
    hira.loc[hira["매칭방식"].isna(), "매칭방식"] = "목록없음"

    # 입점여부: 매칭된 플랫폼 병원의 is_partner. 목록없음 → 0, 보류 → '보류'
    pmap = plat.set_index("id")["is_partner"]
    hira["입점여부"] = hira["플랫폼id"].map(lambda x: pmap.get(int(x)) if pd.notna(x) else pd.NA)
    hira.loc[hira["매칭방식"].isin(["목록없음", "수동(목록없음)"]), "입점여부"] = 0
    hira.loc[hira["매칭방식"] == "보류", "입점여부"] = "보류"
    hira["입점여부"] = hira["입점여부"].astype(str)

    # 플랫폼 쪽에서 매칭되지 않은 병원 (특히 제휴병원은 꼭 직접 확인)
    unmatched_plat = plat[~plat["id"].isin(used)].drop(columns=["key", "base_key", "dong_k"])
    return hira, pd.DataFrame(review), unmatched_plat


# ── 4. 실행 ─────────────────────────────────────────────────────
def main():
    PROC.mkdir(parents=True, exist_ok=True)
    hira = build_seoul_eye()
    plat = pd.read_csv(ROOT / "data/raw/platform_listing.csv")
    df, review, unmatched_plat = match(hira, plat)

    df["확인일"] = CHECKED_DATE
    df["플랫폼병원명"] = df["플랫폼id"].map(lambda x: plat.set_index("id")["병원명"].get(int(x)) if pd.notna(x) else "")
    cols = ["암호화요양기호", "병원명", "종별", "구", "동", "주소", "개설일자", "개원연차", "총의사수",
            "안과전문의수", "수술실보유", "홈페이지보유", "경도", "위도",
            "입점여부", "매칭방식", "플랫폼id", "플랫폼병원명", "확인일"]
    df[cols].to_csv(PROC / "eye_clinics.csv", index=False, encoding="utf-8-sig")
    review.to_csv(PROC / "review_ambiguous.csv", index=False, encoding="utf-8-sig")
    unmatched_plat.to_csv(PROC / "review_unmatched_platform.csv", index=False, encoding="utf-8-sig")

    # ── 요약 ──
    lab = df[df["입점여부"] != "보류"].copy()
    lab["y"] = lab["입점여부"].astype(int)
    print(f"서울 안과: {len(df)}곳 (종별 {df['종별'].value_counts().to_dict()})")
    print(f"매칭방식: {df['매칭방식'].value_counts().to_dict()}")
    print(f"입점 {lab['y'].sum()}곳 / 라벨 확정 {len(lab)}곳 → 입점 비율 {lab['y'].mean():.1%} (보류 {len(df) - len(lab)}곳 제외)")
    print(f"플랫폼 쪽 미매칭: {len(unmatched_plat)}곳 (그중 제휴 {int(unmatched_plat['is_partner'].sum())}곳)")
    print("\n[수술실 보유별 입점 비율]")
    print(lab.groupby("수술실보유")["y"].agg(병원수="count", 입점수="sum", 입점비율="mean").round(3).to_string())
    print("\n[구별 입점 비율] (입점 많은 순)")
    g = lab.groupby("구")["y"].agg(병원수="count", 입점수="sum", 입점비율="mean").round(3)
    print(g.sort_values(["입점수", "병원수"], ascending=False).to_string())


if __name__ == "__main__":
    main()
