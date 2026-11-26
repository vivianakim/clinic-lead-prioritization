"""
03_proposal.py — Top 30 병원 → 1페이지 맞춤 제안서(.md)

Claude Code 스킬(.claude/skills/proposal/SKILL.md)이 이 스크립트를 단계별로 실행한다.
  1) facts   : 숫자를 코드로 계산해 병원별 facts JSON으로 저장 (시작 시각 기록)
  2) (Claude Code가 facts만 보고 문장 JSON을 씀 → outputs/proposals/work/text/rankNN.json)
  3) check   : 문장 JSON 자동 검사 — 형식, facts에 없는 숫자, '(가정)' 누락, 병원명·익명명 노출, 금지 표현
  4) render  : 숫자 + 문장을 합쳐 제안서(원본·익명) 저장, 생성 시간 로그 기록

원칙: 숫자는 코드가, 문장은 AI가. 가정값은 '(가정)'. 환자 1명당 수수료 구조는 제안하지 않음.
     우리가 병원을 고른 기준(개원 연차, 전문의 수, 주변 안과 수)은 내부용이라 제안서에 쓰지 않는다.
     실제 병원에 연락하는 기능은 없다.

사용법
  python -X utf8 src/03_proposal.py facts  --ranks 1 2 3
  python -X utf8 src/03_proposal.py check  --ranks 1 2 3
  python -X utf8 src/03_proposal.py render --ranks 1 2 3
  python -X utf8 src/03_proposal.py template --ranks 1   # (비교용) AI 없이 템플릿 문장 JSON 생성
"""
import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/proposals"                 # 병원명 포함 원본 (git 제외)
WORK = OUT_DIR / "work"                              # facts·text 작업 파일 (git 제외)
PUB_DIR = ROOT / "outputs/public/proposals"          # 익명본
LOG = OUT_DIR / "generation_log.csv"

FORBIDDEN = ["수수료", "알선", "최고", "무조건", "보장", "100%", "1등", "환자 1명당", "건당 지급"]
ASSUMPTION_WORDS = ["광고비", "시술가", "이익률", "손익분기", "공헌이익"]
ALWAYS_OK_NUMBERS = {"1", "2", "3"}                  # '질문 2개', '1페이지' 같은 일반 표현
# 영업 우선순위를 정한 기준(내부용). 병원에 보여 줄 이유가 없고, 평가받는 느낌을 줄 수 있어 제안서 문장에서 막는다.
INTERNAL_WORDS = ["개원", "중앙값", "반경", "전문의"]


def load_module(name, file):
    """파일명이 숫자로 시작하는 스크립트(02_model.py 등)를 모듈로 불러온다."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


roi = load_module("roi", "roi.py")


def fp(kind, rank):
    return WORK / kind / f"rank{rank:02d}.json"


# ── 1. facts: 숫자 계산 ─────────────────────────────────────────
def build_facts(rank, clinics, top):
    r = top[top["순위"] == rank].iloc[0]
    gu = clinics[clinics["구"] == r["구"]]
    me = clinics[clinics["병원명"] == r["병원명"]].iloc[0]
    _, be = roi.break_even()
    return {
        "_meta": {"순위": int(rank), "병원명": r["병원명"], "익명명": r["익명명"],
                  "started_at": datetime.now().isoformat(timespec="seconds")},
        "구": r["구"],
        "구_안과수": int(len(gu)),
        "구_입점수": int(gu["y"].sum()),
        "구_입점비율": round(float(gu["y"].mean()), 3),
        "서울_입점비율": round(float(clinics["y"].mean()), 3),
        # 선정 근거(내부용): 영업 담당자가 참고하는 값. 문장 작성·검사 허용 숫자에서 빠지고 제안서에도 나오지 않는다.
        "_선정근거": {
            "반경1km_안과수": int(me["경쟁밀도_1km"]),
            "구_반경1km_평균": round(float(gu["경쟁밀도_1km"].mean()), 1),
            "개원연차": int(me["개원연차"]),
            "구_개원연차_중앙값": float(gu["개원연차"].median()),
            "안과전문의수": int(me["안과전문의수"]),
            "구_안과전문의_평균": round(float(gu["안과전문의수"].mean()), 1),
            "점수가_높은_이유": r["점수가 높은 이유"],
        },
        "라식라섹_시행": int(me["시력교정신고"]),
        "다초점백내장_신고": int(me["다초점백내장신고"]),
        "라식라섹_시행병원_입점비율": round(float(clinics.loc[clinics["시력교정신고"] == 1, "y"].mean()), 3),
        "추천상품": r["추천 상품(가정)"],
        "ROI_가정": {"월광고비_만원": roi.DEFAULTS["ad_cost"] // 10_000, "라섹_시술가_만원": roi.DEFAULTS["price"] // 10_000,
                    "공헌이익률": roi.DEFAULTS["margin"], "손익분기_월건수": be},
    }


def cmd_facts(ranks):
    model_mod = load_module("model", "02_model.py")
    clinics = model_mod.load_final()
    top = pd.read_csv(ROOT / "outputs/top30.csv")
    (WORK / "facts").mkdir(parents=True, exist_ok=True)
    (WORK / "text").mkdir(parents=True, exist_ok=True)
    for rk in ranks:
        f = build_facts(rk, clinics, top)
        fp("facts", rk).write_text(json.dumps(f, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[1/4 숫자 뽑기] {rk}위 facts 저장: {fp('facts', rk).relative_to(ROOT)}  ({f['_meta']['익명명']})")
    print("→ [2/4 문장 쓰기] 각 facts를 보고 work/text/rankNN.json 작성")


# ── 2. check: 문장 자동 검사 ────────────────────────────────────
def allowed_numbers(f):
    """facts에 있는 숫자와, 비율을 %로 바꾼 값, 반올림 값을 허용 목록으로 만든다."""
    nums = set(ALWAYS_OK_NUMBERS)

    def add(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return
        for x in {v, round(v), round(v, 1)}:
            s = f"{x:g}" if isinstance(x, float) else str(x)
            nums.add(s)
        if 0 < v < 1:                       # 비율 → 퍼센트
            nums.add(str(round(v * 100)))
            nums.add(f"{v * 100:g}")

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if not k.startswith("_"):          # _meta(병원명 등), _선정근거(내부용)는 허용 숫자에서 뺀다
                    walk(v)
        else:
            add(o)

    walk(f)
    return nums


def check_one(rank):
    f = json.loads(fp("facts", rank).read_text(encoding="utf-8"))
    p = fp("text", rank)
    if not p.exists():
        return [f"문장 파일 없음: {p.relative_to(ROOT)}"]
    try:
        t = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        return [f"JSON 형식 오류: {ex}"]
    errs = []
    if set(t) != {"intro", "product_reason", "questions"}:
        errs.append(f"키는 intro, product_reason, questions 세 개여야 함 (현재: {sorted(t)})")
    if not isinstance(t.get("questions"), list) or len(t.get("questions", [])) != 2:
        errs.append("questions는 정확히 2개")
    text = " ".join([str(t.get("intro", "")), str(t.get("product_reason", ""))] + [str(q) for q in t.get("questions", [])])
    ok = allowed_numbers(f)
    bad = sorted({n for n in re.findall(r"\d+(?:\.\d+)?", text) if n not in ok})
    if bad:
        errs.append(f"facts에 없는 숫자: {bad}")
    for name in {f["_meta"]["병원명"], f["_meta"]["익명명"]}:
        if name and name in text:
            errs.append(f"병원명/익명명 노출: '{name}' → '귀원'으로")
    for w in FORBIDDEN:
        if w in text:
            errs.append(f"금지 표현: '{w}'")
    for w in INTERNAL_WORDS:
        if w in text:
            errs.append(f"선정 기준(내부용) 표현: '{w}' — 제안서에는 쓰지 않음")
    for w in ASSUMPTION_WORDS:
        for m in re.finditer(w, text):
            window = text[m.start(): m.start() + 60]
            if re.search(r"\d", window) and "(가정)" not in window:
                errs.append(f"'{w}' 숫자 언급에 '(가정)' 표시 필요")
                break
    return errs


def cmd_check(ranks):
    failed = False
    for rk in ranks:
        errs = check_one(rk)
        if errs:
            failed = True
            print(f"[3/4 자동 검사] {rk}위 실패 → 2/4 문장 쓰기로 돌아가 수정")
            for e in errs:
                print(f"   - {e}")
        else:
            print(f"[3/4 자동 검사] {rk}위 통과")
    sys.exit(1 if failed else 0)


# ── 3. render: 제안서 조립 ──────────────────────────────────────
def render(f, t, display_name):
    roi_md, _ = roi.roi_markdown(procedure="라섹")
    q = t["questions"]
    return f"""# {display_name} 맞춤 제안서

> 개인 분석 · 공개 데이터 기반 (심평원 공개 자료, 2026년 11월 확인). 실제 영업 자료가 아닙니다.

{t['intro']}

## 1. 지역 현황 — {f['구']}

| 항목 | 값 |
| --- | --- |
| {f['구']} 안과 의원 수 | {f['구_안과수']}곳 |
| {f['구']} 플랫폼 입점 안과 | {f['구_입점수']}곳 ({f['구_입점비율']:.0%}, 서울 평균 {f['서울_입점비율']:.0%}) |

## 2. 추천 상품 — {f['추천상품']}

{t['product_reason']}

## 3. ROI — 손익분기 예약 건수

{roi_md}

- 광고는 **정액 기준**으로만 계산했습니다. 환자 1명당 수수료 구조는 제안하지 않습니다.
- 모든 가정값은 실제 광고 단가·시술가·이익률을 넣으면 바로 다시 계산됩니다 (`src/roi.py`).

## 4. 다음 단계 — 미팅에서 확인할 질문

1. {q[0]}
2. {q[1]}
"""


def cmd_render(ranks, mode):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    for rk in ranks:
        errs = check_one(rk)
        if errs:
            print(f"[4/4 조립·기록] {rk}위 검사 실패로 조립하지 않음: {errs}")
            continue
        f = json.loads(fp("facts", rk).read_text(encoding="utf-8"))
        t = json.loads(fp("text", rk).read_text(encoding="utf-8"))
        meta = f["_meta"]
        stem = f"rank{rk:02d}"
        (OUT_DIR / f"{stem}_{meta['병원명']}.md").write_text(render(f, t, meta["병원명"]), encoding="utf-8")
        public = render(f, t, meta["익명명"]).replace(meta["병원명"], meta["익명명"])
        (PUB_DIR / f"{stem}_{meta['익명명'].replace(' ', '_')}.md").write_text(public, encoding="utf-8")
        start = datetime.fromisoformat(meta["started_at"])
        sec = round((datetime.now() - start).total_seconds())
        row = {"완료시각": datetime.now().isoformat(timespec="seconds"), "순위": rk, "익명명": meta["익명명"],
               "모드": mode, "facts부터_완료까지_초": sec}
        pd.DataFrame([row]).to_csv(LOG, mode="a", index=False, header=not LOG.exists(), encoding="utf-8-sig")
        print(f"[4/4 조립·기록] {rk}위 {meta['익명명']} 제안서 저장 (모드={mode}, facts부터 {sec}초)")


# ── (비교용) 템플릿 문장 ─────────────────────────────────────────
def cmd_template(ranks):
    for rk in ranks:
        f = json.loads(fp("facts", rk).read_text(encoding="utf-8"))
        intro = (f"귀원이 있는 {f['구']}에는 안과 의원 {f['구_안과수']}곳이 있고, 그중 플랫폼에 입점한 곳은 {f['구_입점수']}곳입니다. "
                 "귀원이 이 지역에서 환자에게 먼저 보이도록 돕는 방법을 제안드립니다.")
        if f["라식라섹_시행"]:
            reason = ("귀원은 라식·라섹 비급여 가격을 공개하고 있어, 가격을 비교하는 환자에게 바로 보이는 시술 할인 이벤트 광고와 잘 맞습니다. "
                      f"서울에서 라식·라섹을 시행하는 안과 의원 중 {round(f['라식라섹_시행병원_입점비율'] * 100)}%가 이미 플랫폼에 입점해 있습니다.")
        elif f["다초점백내장_신고"]:
            reason = "귀원은 다초점 백내장 렌즈 가격을 공개하고 있어, 상담 전에 비용을 알아보는 환자를 상담·견적 연결로 이어 주기 좋습니다."
        else:
            reason = "검진·일반 진료 할인으로 지역 환자에게 귀원을 처음 알리는 데 검진·진료 할인 상품이 부담이 적습니다."
        t = {"intro": intro, "product_reason": reason,
             "questions": ["현재 신규 환자는 주로 어떤 경로(검색, 소개, 광고)로 오시나요?",
                           "가장 늘리고 싶은 시술과 월 목표 건수가 있으신가요?"]}
        fp("text", rk).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"템플릿 문장 저장: {fp('text', rk).relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["facts", "check", "render", "template"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--mode", default="claude-code", help="render 로그에 남길 문장 작성 방식")
    a = ap.parse_args()
    {"facts": lambda: cmd_facts(a.ranks), "check": lambda: cmd_check(a.ranks),
     "render": lambda: cmd_render(a.ranks, a.mode), "template": lambda: cmd_template(a.ranks)}[a.step]()
