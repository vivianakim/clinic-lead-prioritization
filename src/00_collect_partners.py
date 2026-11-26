"""
00_collect_partners.py — 병원 플랫폼의 서울 안과 공개 목록 정리 (정답 라벨용)

[수집 방법 요약]
플랫폼의 API를 직접 부르지 않았다. 브라우저에서 공개 목록 페이지를 열고 '20개 더보기' 버튼을 3초 간격으로 눌렀다.
  <플랫폼의 서울 안과 공개 목록 페이지 URL>  (이용약관·robots.txt 확인 후 사용)
- 1페이지: 페이지 HTML에 이미 들어 있는 데이터(__NEXT_DATA__)에서 읽음
- 2페이지~: 버튼 클릭으로 페이지가 받아 온 응답을 페이지 안에서 읽음 (새 요청을 만들지 않음)
- 저장한 항목: id, 병원명, 주소(시·구·동), is_partner, 페이지 번호 — 리뷰·가격·이미지는 저장하지 않음
- 차단 신호(200이 아닌 응답, 빈 응답)가 나오면 즉시 멈추도록 했고, 실제로는 29페이지 모두 200
- 브라우저 코드: src/collect_browser_snippet.js
- 결과 원본: data/raw/platform_capture_20261124.tsv (1차, '방문 많은 순', 574곳)
- 1차는 클릭 사이 정렬이 바뀌어 4곳이 빠졌다. 그래서 '리뷰 많은 순'으로 같은 방식으로 한 번 더 수집(2차, 572곳)했고,
  두 결과를 합치면 정확히 578곳이 된다. 2차에서만 나온 4곳: data/raw/platform_capture_20261124_pass2_extra.tsv
  (두 번 모두 제휴 91곳으로 같았다)

[이 스크립트가 하는 일]
원본 TSV를 읽어 구·동을 나누고, 중복을 확인한 뒤 data/raw/platform_listing.csv로 저장한다.
"""
from pathlib import Path

import pandas as pd

# ── 경로와 수집 정보 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CAPTURES = [ROOT / "data/raw/platform_capture_20261124.tsv",            # 1차: 방문 많은 순
            ROOT / "data/raw/platform_capture_20261124_pass2_extra.tsv"]  # 2차: 리뷰 많은 순에서만 나온 병원
OUT = ROOT / "data/raw/platform_listing.csv"
CHECKED_DATE = "2026-11-24"   # 목록을 확인한 날짜
SITE_TOTAL = 578              # 화면(응답)에 표시된 서울 안과 수


def main():
    # 1) 원본 읽기 (1차 + 2차 보충분을 합침. id는 숫자, is_partner는 true/false 문자열)
    df = pd.concat([pd.read_csv(p, sep="\t", dtype={"id": int}) for p in CAPTURES], ignore_index=True)
    df["is_partner"] = df["is_partner"].astype(str).str.lower().eq("true").astype(int)

    # 2) 주소 '서울 강남구 역삼1동' → 시, 구, 동으로 나누기
    parts = df["address"].str.split(" ", n=2, expand=True)
    df["시"], df["구"], df["동"] = parts[0], parts[1], parts[2]

    # 3) 중복 확인: 같은 id가 두 번 들어오면 하나만 남김
    dup = df["id"].duplicated().sum()
    df = df.drop_duplicates("id")

    # 4) 저장할 컬럼만 남김 (병원명·지역·제휴여부)
    out = df[["id", "name", "시", "구", "동", "is_partner"]].rename(columns={"name": "병원명"})
    out["확인일"] = CHECKED_DATE
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    # 5) 요약 출력
    print(f"수집 병원 수: {len(out)} (사이트 표시 {SITE_TOTAL}, 차이 {SITE_TOTAL - len(out)})")
    print(f"제휴(is_partner=1): {out['is_partner'].sum()}곳, 중복 제거: {dup}건")
    print(f"저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
