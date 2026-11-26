// 병원 플랫폼의 서울 안과 공개 목록 수집용 브라우저 코드 (틀)
// 2026-11-24 한 플랫폼의 공개 목록 페이지에서 쓴 코드를, 대상 사이트 고유의 이름만 빼고 일반화했다.
// 사용법: 대상 사이트의 목록 페이지 구조를 먼저 확인해 아래 SITE 설정을 채운 뒤,
//         개발자 도구 콘솔에 붙여 넣고 마지막 줄의 __clickBatch(10)을 여러 번 실행한다.
// 원칙: 이용약관·robots.txt 확인 후 사용. API를 직접 부르지 않고, 버튼 클릭으로 페이지가 받아 온 응답만 읽는다.
//       id·병원명·주소·제휴 여부만 보관하고, 3초 간격으로 누르며, 200이 아닌 응답이 오면 멈춘다.

// ── 대상 사이트 설정 (페이지 구조를 확인한 뒤 채우기) ─────────────────
const SITE = {
  ssrScriptId: "__NEXT_DATA__",         // 1페이지 데이터가 들어 있는 <script> id (Next.js 사이트의 기본값)
  ssrListKey: "<1페이지 병원 목록 키>",   // 1페이지 데이터 안에서 병원 배열이 들어 있는 키
  ssrTotalKey: "<1페이지 전체 병원 수 키>",
  moreRequestPath: "<더보기 요청 주소에 들어가는 경로 일부>",  // '더보기' 클릭 때 페이지가 보내는 요청을 알아볼 문자열
  moreListKey: "<더보기 응답의 병원 목록 키>",
  moreTotalKey: "<더보기 응답의 전체 병원 수 키>",
  moreButtonText: "20개 더보기",          // 화면에 보이는 '더보기' 버튼 글자
};

window.__collect = { rows: {}, log: [] };
const keep = (h, page) => {
  window.__collect.rows[h.id] = { id: h.id, name: h.name, address: h.address, is_partner: (h.is_partner ?? h.isPartner), page };
};

// 1페이지: 페이지 HTML 안에 미리 들어 있는 데이터에서 읽기
(function walk(o) {
  if (!o) return;
  if (typeof o === "string" && o.includes(SITE.ssrListKey)) { try { walk(JSON.parse(o)); } catch (e) {} return; }
  if (typeof o !== "object") return;
  if (Array.isArray(o[SITE.ssrListKey])) {
    o[SITE.ssrListKey].forEach(h => keep(h, 1));
    window.__collect.log.push({ page: 1, status: "ssr", n: o[SITE.ssrListKey].length, total: o[SITE.ssrTotalKey] });
    return;
  }
  for (const k in o) walk(o[k]);
})(JSON.parse(document.getElementById(SITE.ssrScriptId).textContent));

// 2페이지~: '더보기' 클릭 때 오는 응답에서 같은 항목만 읽기
const _open = XMLHttpRequest.prototype.open, _send = XMLHttpRequest.prototype.send;
XMLHttpRequest.prototype.open = function (m, u) { this.__u = u; return _open.apply(this, arguments); };
XMLHttpRequest.prototype.send = function () {
  if ((this.__u || "").includes(SITE.moreRequestPath)) {
    this.addEventListener("loadend", () => {
      const pg = +(new URL(this.__u, location.href).searchParams.get("page"));
      let n = 0, tot = null;
      try { const j = JSON.parse(this.responseText); tot = j[SITE.moreTotalKey]; (j[SITE.moreListKey] || []).forEach(h => { keep(h, pg); n++; }); } catch (e) {}
      window.__collect.log.push({ page: pg, status: this.status, n, total: tot });
    });
  }
  return _send.apply(this, arguments);
};

// 버튼을 3초 간격으로 누르고, 응답이 없거나 200이 아니면 멈춤
window.__clickBatch = async (maxClicks) => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  for (let k = 0; k < maxClicks; k++) {
    const cands = [...document.querySelectorAll("*")].filter(e => (e.innerText || "").trim() === SITE.moreButtonText);
    if (!cands.length) return { stop: "no button" };
    const el = cands[cands.length - 1]; el.scrollIntoView({ block: "center" }); await sleep(300);
    const rc = el.getBoundingClientRect();
    const before = window.__collect.log.length;
    document.elementFromPoint(rc.left + rc.width / 2, rc.top + rc.height / 2).click();
    await sleep(3000);
    if (window.__collect.log.length === before) return { stop: "no response" };
    const last = window.__collect.log[window.__collect.log.length - 1];
    if (last.status !== 200 || last.n === 0) return { stop: "bad", last };
  }
  return { stop: "batch done" };
};
// await __clickBatch(10);
