r"""03_check_demo.py — 제안서 자동 검사 시연 (포트폴리오 페이지의 "검사가 실제로 잡아내는 것")
일부러 잘못 쓴 문장을 임시 파일에 넣고 03_proposal.py의 check_one으로 검사한다. 실제 작업 파일은 건드리지 않는다.
실행: .venv\Scripts\python -X utf8 src\03_check_demo.py   (먼저 03_proposal.py facts --ranks 1 필요)
"""
import importlib.util, json, pathlib, tempfile, sys
spec = importlib.util.spec_from_file_location("p", str(pathlib.Path(__file__).with_name("03_proposal.py"))); m = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(m)
bad = {"intro": "강북구 A안과는 강북구 최고의 안과로, 입점하면 신규 환자가 30% 늘어납니다.",
       "product_reason": "월 광고비 100만원이면 충분히 회수됩니다.",
       "questions": ["현재 신규 환자는 어떤 경로로 오시나요?", "늘리고 싶은 시술은 무엇인가요?"]}
tmp = pathlib.Path(tempfile.mkdtemp()) / "text.json"; tmp.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
orig = m.fp
m.fp = lambda kind, rank: tmp if kind == "text" else orig(kind, rank)
for e in m.check_one(1): print("-", e)
print("FORBIDDEN:", m.FORBIDDEN); print("ASSUMPTION:", m.ASSUMPTION_WORDS)
