"""
mvp_ver_1_9_3.py — CRISIS CONSULTANT v1.9.3 (NATAM 위기관리 + 저작권 침해 통합)

v1.9.2(크리에이터 위기/논란 분석)와 copyright_detector(저작권 침해 자동 감지)를
하나의 입력 흐름으로 묶은 통합 실행기.

┌──────────────────────────────────────────────────────────────────────┐
│ 하나의 영상/URL 입력 → ① NATAM 위기 분석  +  ② 저작권 침해 분석  둘 다 수행 │
│ → 두 결과를 합친 통합 리포트를 MD · PDF · JSON 3종으로 모두 생성            │
└──────────────────────────────────────────────────────────────────────┘

설계 요약
  · NATAM 쪽(v1.9.2)은 통째로 '모듈로 import' 해서 재사용한다(코드 복제 없음).
  · 저작권 쪽(copyright_detector)은 자체 .env / sqlite / 상대경로 가정이 있어
    '독립 서브프로세스'로 실행한다(cwd = copyright_detector).
  · 리포트는 NATAM 분석 직후 곧바로 만들지 않는다. 저작권 분석까지 끝난 뒤
    두 결과를 병합해 '통합 템플릿(report_template_v1_9_3.md)' 기반으로
    MD·PDF·JSON 3종을 한 번에 생성한다.
      - analyze_video_full 내부의 조기 리포트 생성은 잠시 억제(monkeypatch)하고
        분석 결과(report)만 받아온 뒤, 저작권을 합쳐 최종 리포트를 만든다.

사용법:  python mvp_ver_1_9_3.py     (실행 후 URL/파일경로/텍스트 입력)
"""
import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path

# ── 항상 이 스크립트(=NATAM) 폴더를 작업 디렉터리로 고정 ──────────────
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# ── v1.9.2 를 모듈로 가져와 전부 재사용 ───────────────────────────────
import mvp_ver_1_9_2 as mvp

# ── 저작권 감지기 경로 ────────────────────────────────────────────────
COPYRIGHT_DIR     = SCRIPT_DIR / "copyright_detector"
COPYRIGHT_MAIN    = COPYRIGHT_DIR / "main.py"
COPYRIGHT_RESULTS = COPYRIGHT_DIR / "results"

# ── 통합 리포트 템플릿 (NATAM 섹션 + 저작권 섹션8) ───────────────────
V193_TEMPLATE = "report_template_v1_9_3.md"

# 표시용 라벨/이모지
_CR_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "SAFE": "✅"}
_CR_TYPE_LABEL = {
    "music": "🎵 음악", "video_clip": "🎬 영상클립", "image": "🖼️ 이미지",
    "logo": "🏷️ 로고", "font": "🔤 폰트",
}


# ════════════════════════════════════════════════════════════════════
# 저작권 분석 (copyright_detector 를 독립 프로세스로 실행)
# ════════════════════════════════════════════════════════════════════
def run_copyright_detection(video_path: str, force: bool = False) -> dict | None:
    """
    로컬 영상 파일에 대해 copyright_detector/main.py 를 서브프로세스로 실행한다.

    · cwd = copyright_detector → 자체 .env / sqlite / 상대경로가 그대로 동작.
    · 진행 로그·요약은 서브프로세스가 직접 콘솔에 출력(실시간 확인 가능).
    · 저장된 결과 JSON을 읽어 통합 리포트용 dict 로 반환. 실패 시 None.
    """
    if not COPYRIGHT_MAIN.exists():
        print(f"⚠️  copyright_detector/main.py 를 찾을 수 없어 저작권 분석을 건너뜁니다.\n"
              f"    (예상 위치: {COPYRIGHT_MAIN})")
        return None

    abs_video = os.path.abspath(video_path)
    if not os.path.exists(abs_video):
        print(f"⚠️  저작권 분석용 영상 파일을 찾을 수 없어 건너뜁니다: {abs_video}")
        return None

    COPYRIGHT_RESULTS.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    cmd = [sys.executable, "main.py", abs_video, "--format", "json"]
    if force:
        cmd.append("--force")

    print("\n" + "=" * 63)
    print("©️   저작권 침해 분석 시작  (copyright_detector)")
    print(f"     대상: {os.path.basename(abs_video)}")
    print("=" * 63)

    try:
        proc = subprocess.run(cmd, cwd=str(COPYRIGHT_DIR), check=False)
        if proc.returncode != 0:
            print(f"⚠️  저작권 분석 프로세스가 비정상 종료(코드 {proc.returncode}).")
    except Exception as e:
        print(f"⚠️  저작권 분석 실행 실패: {e}")
        return None

    # 이번 실행에서 새로 생성된 결과 JSON 을 mtime 기준으로 탐색
    latest, latest_m = None, started_at - 1.0
    for p in COPYRIGHT_RESULTS.glob("result_*.json"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= started_at and m > latest_m:
            latest, latest_m = p, m
    if latest is None:  # 폴백: 가장 최근 결과라도 사용
        cands = sorted(COPYRIGHT_RESULTS.glob("result_*.json"),
                       key=lambda p: p.stat().st_mtime)
        latest = cands[-1] if cands else None

    if latest is None:
        print("⚠️  저작권 결과 JSON 을 찾지 못했습니다(요약은 위 로그 참조).")
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  저작권 결과 JSON 파싱 실패({e}) — 요약은 위 로그 참조.")
        return None


# ════════════════════════════════════════════════════════════════════
# 저작권 결과 → MD 템플릿 플레이스홀더({CR_*})
# ════════════════════════════════════════════════════════════════════
def build_copyright_placeholders(cr: dict | None) -> dict:
    """copyright_detector 결과 dict 를 report_template_v1_9_3.md 의 {CR_*} 로 변환."""
    if not cr:
        return {
            "CR_STATUS": "⚠️ 저작권 분석 미수행 또는 실패 (아래는 데이터 없음)",
            "CR_OVERALL_EMOJI": "",
            "CR_OVERALL_LEVEL": "N/A",
            "CR_OVERALL_SCORE": "0",
            "CR_TOTAL_ISSUES": "0",
            "CR_HIGH_COUNT": "0",
            "CR_MEDIUM_COUNT": "0",
            "CR_VIDEO_DURATION": "—",
            "CR_YOUTUBE_BLOCK": "> 저작권 분석 결과가 없어 유튜브 예측을 표시할 수 없습니다.",
            "CR_BYTYPE_ROWS": "| (데이터 없음) | — |",
            "CR_TIMELINE_ROWS": "| — | — | 저작권 분석 결과 없음 | — | — |",
        }

    s      = cr.get("summary", {})
    level  = s.get("overall_risk_level", "SAFE")
    score  = s.get("overall_risk_score", 0)
    dur    = cr.get("video_duration", 0)
    dur_s  = f"{dur:.0f}초" if isinstance(dur, (int, float)) else str(dur)

    # 분류별 건수
    by_type = s.get("by_type", {})
    if by_type:
        bytype_rows = "\n".join(
            f"| {_CR_TYPE_LABEL.get(t, t)} | {c}건 |" for t, c in by_type.items())
    else:
        bytype_rows = "| (탐지된 항목 없음) | 0건 |"

    # 유튜브 예측 블록
    yt = s.get("youtube")
    if yt:
        yt_block = (
            f"> **{yt.get('headline', '')}**\n>\n"
            f"> - 수익화 영향(노란 딱지): **{yt.get('monetization_impact', '-')}**\n"
            f"> - Content ID 클레임 확률: **{yt.get('claim_probability', '-')}%**\n"
            f"> - 차단 위험: {yt.get('block_risk', '-')}  |  "
            f"저작권 경고(Strike) 위험: {yt.get('strike_risk', '-')}"
        )
        if yt.get("advice"):
            yt_block += f"\n> - 💡 {yt.get('advice')}"
    else:
        yt_block = "> 유튜브 예측 데이터가 없습니다(위험 항목 미검출)."

    # 타임라인 (HIGH/MEDIUM, 위험도 높은 순 상위 15)
    def _num(f):
        try:
            return float(str(f.get("risk_score", "0")).replace("%", ""))
        except (ValueError, TypeError):
            return 0.0

    risky = [f for f in cr.get("timeline", [])
             if f.get("risk_level") in ("HIGH", "MEDIUM")]
    risky.sort(key=lambda f: -_num(f))
    rows = []
    for f in risky[:15]:
        title = str(f.get("title") or f.get("rights_holder") or "-").replace("|", "/")[:40]
        src   = str(f.get("rights_holder") or f.get("source") or "").replace("|", "/")[:20]
        disp  = title + (f" ({src})" if src and src not in title else "")
        lv    = f.get("risk_level", "")
        rows.append(
            f"| {f.get('timestamp', '00:00')} "
            f"| {_CR_TYPE_LABEL.get(f.get('type', ''), f.get('type', ''))} "
            f"| {disp} "
            f"| {_CR_EMOJI.get(lv, '')} {lv} {f.get('risk_score', '')} "
            f"| {f.get('yt_emoji', '')} {f.get('yt_outcome_label', '') or '-'} |"
        )
    if not rows:
        rows = ["| — | — | 🔴/🟡 위험 등급 항목 없음 | ✅ SAFE | — |"]

    return {
        "CR_STATUS": "✅ 저작권 분석 완료",
        "CR_OVERALL_EMOJI": _CR_EMOJI.get(level, ""),
        "CR_OVERALL_LEVEL": level,
        "CR_OVERALL_SCORE": str(score),
        "CR_TOTAL_ISSUES": str(s.get("total_issues_found", 0)),
        "CR_HIGH_COUNT": str(s.get("high_risk_count", 0)),
        "CR_MEDIUM_COUNT": str(s.get("medium_risk_count", 0)),
        "CR_VIDEO_DURATION": dur_s,
        "CR_YOUTUBE_BLOCK": yt_block,
        "CR_BYTYPE_ROWS": bytype_rows,
        "CR_TIMELINE_ROWS": "\n".join(rows),
    }


# ── 저작권 섹션(템플릿 8장) 마커: 저작권 미수행 시 이 구간을 통째로 제거 ──
_CR_SECTION_START = "<!-- COPYRIGHT_SECTION_START -->"
_CR_SECTION_END   = "<!-- COPYRIGHT_SECTION_END -->"


def _apply_copyright_section(md_text: str, include: bool) -> str:
    """
    통합 MD 템플릿에서 저작권 섹션(마커로 감싼 8장)을 처리한다.
      · include=True  → 마커 주석 줄만 제거(섹션 내용은 그대로 유지)
      · include=False → 마커 사이(포함) 전체 제거(저작권 섹션 완전 삭제)
    """
    if include:
        return re.sub(r'[ \t]*<!-- COPYRIGHT_SECTION_(?:START|END) -->[ \t]*\n?',
                      '', md_text)
    return re.sub(
        re.escape(_CR_SECTION_START) + r'.*?' + re.escape(_CR_SECTION_END) + r'\n?',
        '', md_text, flags=re.DOTALL)


# ════════════════════════════════════════════════════════════════════
# 통합 MD 리포트 엔진 (NATAM 엔진 재사용 + 저작권 플레이스홀더 추가)
# ════════════════════════════════════════════════════════════════════
class CrisisReportEngineV193(mvp.CrisisReportEngine):
    """NATAM 리포트 엔진을 상속해 {CR_*} 저작권 플레이스홀더를 얹은 통합 엔진."""

    def __init__(self, template_path: str = V193_TEMPLATE, output_dir: str = "reports",
                 include_copyright: bool = True):
        super().__init__(template_path=template_path, output_dir=output_dir)
        self.include_copyright = include_copyright

    def _load_template(self) -> str | None:
        # 템플릿을 읽은 뒤, 저작권 미수행이면 8장 섹션을 통째로 제거한다.
        content = super()._load_template()
        if content is None:
            return None
        return _apply_copyright_section(content, self.include_copyright)

    def _build_data_map(self, report: dict) -> dict:
        data_map = super()._build_data_map(report)          # NATAM 전체 플레이스홀더
        data_map.update(build_copyright_placeholders(report.get("copyright")))
        return data_map


# ════════════════════════════════════════════════════════════════════
# NATAM 분석 실행 (내부 조기 리포트 생성은 억제) + 통합 리포트 생성
# ════════════════════════════════════════════════════════════════════
def analyze_only(system, video_input: str, download_dir: str | None = None):
    """
    analyze_video_full 을 실행하되 '내부 MD/PDF 생성'을 잠시 억제한다.
    저작권 결과를 합쳐 최종 리포트를 만들어야 하므로, 여기서는 분석 결과(report)만 받는다.
    (내부 JSON은 저작권 없이 한 번 저장되지만, finalize 단계에서 덮어쓴다.)
    """
    orig_create  = system.report_engine.create_report
    orig_gen_pdf = getattr(mvp, "_gen_pdf", None)
    system.report_engine.create_report = lambda report: None       # MD 조기생성 억제
    if orig_gen_pdf is not None:
        mvp._gen_pdf = lambda report, output_dir="reports": None    # PDF 조기생성 억제
    try:
        kwargs = dict(video_input=video_input,
                      output_dir="samples/transcripts", use_cache=True)
        if download_dir:
            kwargs["download_dir"] = download_dir
        report, json_path, _md, _pdf = system.analyze_video_full(**kwargs)
    finally:
        system.report_engine.create_report = orig_create
        if orig_gen_pdf is not None:
            mvp._gen_pdf = orig_gen_pdf
    return report, json_path


def finalize_reports(report: dict, json_path: str, copyright_results: dict | None,
                     include_copyright: bool = True):
    """
    NATAM report 에 저작권 결과를 병합하고 JSON·MD·PDF 3종을 모두 생성한다.
      · include_copyright=True  → copyright 키를 넣어 저작권 섹션을 렌더
        (copyright_results 가 None 이면 '미수행/실패'로 표기)
      · include_copyright=False → copyright 키를 아예 제거해 JSON·PDF·MD 세 리포트
        모두에서 저작권 섹션을 완전히 제외(사용자가 저작권 분석을 건너뛴 경우)
    """
    if include_copyright:
        report["copyright"] = copyright_results   # None 이어도 키를 넣어 섹션을 렌더
    else:
        report.pop("copyright", None)             # 키 제거 → 세 리포트 모두 저작권 섹션 제외

    print("\n📄 통합 리포트(위기 + 저작권) 생성 중..." if include_copyright
          else "\n📄 위기 분석 리포트 생성 중... (저작권 섹션 제외)")

    # ── JSON (include_copyright=False 면 copyright 키 없이 저장) ──
    mvp._save_json(report, json_path)

    # ── MD (통합 템플릿; 저작권 미수행 시 8장 섹션 제거) ──
    md_path = CrisisReportEngineV193(include_copyright=include_copyright).create_report(report)

    # ── PDF (report 에 copyright 키가 있을 때만 저작권 페이지 렌더) ──
    pdf_path = None
    gen_pdf = getattr(mvp, "_gen_pdf", None)
    if getattr(mvp, "_PDF_AVAILABLE", False) and gen_pdf:
        try:
            pdf_path = gen_pdf(report, output_dir="reports")
        except Exception as e:
            print(f"⚠️ PDF 생성 실패: {e}")
    else:
        print("⚠️ pdf_report_generator(reportlab) 미사용 → PDF 생성 건너뜀")

    return md_path, pdf_path


# ════════════════════════════════════════════════════════════════════
# 통합 요약 배너
# ════════════════════════════════════════════════════════════════════
def print_combined_banner(natam_report: dict | None, copyright_results: dict | None,
                          copyright_requested: bool = True):
    """NATAM 위기 결과와 저작권 결과를 한눈에 보이는 통합 요약."""
    print("\n" + "█" * 63)
    print("  🧩  통합 분석 요약  (NATAM 위기관리  +  저작권 침해)")
    print("█" * 63)

    natam = (natam_report or {}).get("natam_risk") or {}
    meta  = (natam_report or {}).get("meta") or {}
    title = meta.get("incident_title") or meta.get("video_filename") or "-"
    print(f"  🛡️  NATAM 위기관리")
    print(f"       사건: {str(title)[:50]}")
    if natam:
        print(f"       A축 종합: {natam.get('overall_a', '?')}   |   "
              f"B축 종합: {natam.get('overall_b', '?')}")
    else:
        print(f"       (NATAM 리스크 결과 없음)")

    print(f"  ©️   저작권 침해")
    if copyright_results:
        s = copyright_results.get("summary", {})
        level = s.get("overall_risk_level", "?")
        score = s.get("overall_risk_score", 0)
        total = s.get("total_issues_found", 0)
        emoji = _CR_EMOJI.get(level, "")
        print(f"       전체 위험도: {emoji} {level} — {score}%   (발견 {total}건)")
        by_type = s.get("by_type", {})
        if by_type:
            parts = [f"{_CR_TYPE_LABEL.get(t, t)} {c}" for t, c in by_type.items()]
            print(f"       분류별: {' · '.join(parts)}")
        yt = s.get("youtube")
        if yt:
            print(f"       📺 유튜브 예측: 수익화영향 {yt.get('monetization_impact')}  |  "
                  f"Content ID 클레임확률 {yt.get('claim_probability')}%")
    elif not copyright_requested:
        print(f"       (사용자 선택으로 저작권 분석을 건너뜀 — 리포트에서 제외됨)")
    else:
        print(f"       (결과 없음 / 실패 — 위 저작권 로그 참조)")

    print("█" * 63 + "\n")


def _print_report_paths(json_path, md_path, pdf_path):
    print(f"\n✅ 통합 JSON 보고서 : {json_path}")
    if md_path:  print(f"✅ 통합 MD  리포트  : {md_path}")
    if pdf_path: print(f"✅ 통합 PDF 리포트  : {pdf_path}")


# ════════════════════════════════════════════════════════════════════
# 저작권 분석 수행 여부 선택 (URL·영상 입력 직후 물어봄)
# ════════════════════════════════════════════════════════════════════
def ask_copyright_choice() -> bool:
    """
    URL·영상 입력 후, 저작권 침해 분석까지 함께 수행할지 물어본다.
      · [Enter] 또는 y  → 위기 분석 + 저작권 침해 분석 (기본값)
      · n               → NATAM 위기 분석만
    """
    ans = input("   ©️  저작권 침해 분석도 함께 수행할까요?  "
                "[Y] 위기+저작권(기본)  /  [n] 위기 분석만 : ").strip().lower()
    do_cr = ans not in ("n", "no", "ㄴ", "아니오", "아니", "x")
    print("   → 위기 분석 + 저작권 침해 분석을 수행합니다." if do_cr
          else "   → NATAM 위기 분석만 수행합니다.")
    return do_cr


# ════════════════════════════════════════════════════════════════════
# 메인 루프
# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 63)
    print("🛡️©️  CRISIS CONSULTANT v1.9.3 — 위기관리 + 저작권 통합 분석기")
    print("     NATAM v2.0 위기 분석  ⨉  copyright_detector 저작권 침해 감지")
    print("     리포트: 통합 MD · PDF · JSON 3종 동시 생성")
    print("=" * 63)

    system = mvp.CrisisConsultantSystem(
        db_path            = 'case_db.json',
        rules_yaml         = 'rules.yaml',
        labels_yaml        = 'controversy_labels.yaml',
        worst_actions_yaml = 'worst_actions_map.yaml',
        template_path      = 'report template.md',
        report_dir         = 'reports',
    )

    VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4a', '.mp3', '.wav'}

    while True:
        print("\n" + "─" * 63)
        print("유튜브 URL 입력      →  NATAM 위기 분석  (저작권 분석 함께할지 선택)")
        print("영상 파일 경로 입력  →  NATAM 위기 분석  (저작권 분석 함께할지 선택)")
        print("텍스트 입력          →  NATAM 위기 분석만 (영상이 없어 저작권 분석 불가)")
        print("[q]                  →  종료")
        print("   ↳ URL·영상 입력 후 [Y] 위기+저작권 / [n] 위기 분석만 을 고릅니다.")
        user_input = input("💬 입력: ").strip().replace('"', '')

        if user_input.lower() == 'q':
            print("👋 종료합니다.")
            break
        if not user_input:
            continue

        # ── 유튜브 URL ──────────────────────────────────────
        if mvp.is_youtube_url(user_input):
            do_copyright = ask_copyright_choice()
            report, json_path = analyze_only(system, user_input, download_dir="downloads")
            mvp.print_report(report)

            copyright_results = None
            if do_copyright:
                local_video = (report.get("youtube_meta") or {}).get("video_path")
                if local_video:
                    copyright_results = run_copyright_detection(local_video)
                else:
                    print("⚠️  다운로드된 영상 경로를 찾을 수 없어 저작권 분석을 건너뜁니다.")

            md_path, pdf_path = finalize_reports(report, json_path, copyright_results,
                                                 include_copyright=do_copyright)
            _print_report_paths(json_path, md_path, pdf_path)
            print_combined_banner(report, copyright_results, copyright_requested=do_copyright)

        # ── Google Drive URL ────────────────────────────────
        elif mvp.is_google_drive_url(user_input):
            do_copyright = ask_copyright_choice()
            # 한 번만 내려받아 두 분석이 같은 파일을 쓰도록 미리 다운로드
            dl = mvp.download_google_drive_video(user_input, "downloads")
            local_video = dl['video_path']
            report, json_path = analyze_only(system, local_video)
            mvp.print_report(report)

            copyright_results = run_copyright_detection(local_video) if do_copyright else None
            md_path, pdf_path = finalize_reports(report, json_path, copyright_results,
                                                 include_copyright=do_copyright)
            _print_report_paths(json_path, md_path, pdf_path)
            print_combined_banner(report, copyright_results, copyright_requested=do_copyright)

        # ── 로컬 영상 파일 ──────────────────────────────────
        elif os.path.splitext(user_input)[1].lower() in VIDEO_EXTS:
            if not os.path.exists(user_input):
                print("❌ 파일을 찾을 수 없습니다.")
                continue
            do_copyright = ask_copyright_choice()
            report, json_path = analyze_only(system, user_input)
            mvp.print_report(report)

            copyright_results = run_copyright_detection(user_input) if do_copyright else None
            md_path, pdf_path = finalize_reports(report, json_path, copyright_results,
                                                 include_copyright=do_copyright)
            _print_report_paths(json_path, md_path, pdf_path)
            print_combined_banner(report, copyright_results, copyright_requested=do_copyright)

        # ── 텍스트 직접 입력 (영상 없음 → NATAM 분석만) ─────
        else:
            query       = user_input
            rule_result = system.rule_scan(query)
            if rule_result["hit"]:
                print(f"\n🚨 [룰 적발] {rule_result['policy']} ({rule_result['severity']}) "
                      f"→ {rule_result['action']}  원인어: '{rule_result['matched_word']}'")

            print("\n⏳ 유사 사례 검색 중...")
            cases, dists, summary = system.search_and_analyze(query)
            print("⏳ 논란 유형 분류 중...")
            classification = system.classify_controversy(query)

            spread_result = {"stage": "Early", "reasons": [], "metrics": None}
            if cases and cases[0].get("incident_metadata"):
                spread_result = system.spread_analyzer.predict_stage(cases[0])
            worst_actions = system.get_risk_guide(
                cases[0].get("controversy_type", "") if cases else "",
                spread_result["stage"].lower(),
            )

            print("⏳ NATAM v2.0 리스크 평가 중...")
            natam_result = mvp.assess_natam_risk(
                client             = system.client,
                system_instruction = system.system_instruction,
                transcript_text    = "",
                summary            = query,
                gen_model          = mvp.GEN_MODEL,
            )
            print(f"   A축 종합: {natam_result['overall_a']} | B축 종합: {natam_result['overall_b']}")

            ts     = mvp.datetime.now().strftime("%Y%m%d_%H%M%S")
            report = {
                "meta": {
                    "input_query": query,
                    "analyzed_at": mvp.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type":        "text_input",
                },
                "rule_scan":           rule_result,
                "spread_stage":        spread_result,
                "classification":      classification,
                "similar_cases": [
                    {
                        "rank":             i + 1,
                        "title":            c["title"],
                        "controversy_type": c.get("controversy_type", ""),
                        "distance":         round(float(d), 4),
                        "response_pattern": c.get("response_pattern", []),
                    }
                    for i, (c, d) in enumerate(zip(cases, dists))
                ],
                "pattern_summary":     summary.strip(),
                "worst_actions":       worst_actions,
                "transcript_analysis": [],
                "keyframe_analysis":   [],
                "transcript_files":    {},
                "youtube_meta":        {},
                "natam_risk":          natam_result,
                # 텍스트 입력은 영상이 없어 저작권 분석 미수행 → copyright 키를 아예 넣지 않아
                # JSON·MD 리포트에서 저작권 섹션을 완전히 제외한다.
            }

            os.makedirs("reports", exist_ok=True)
            json_path = os.path.join("reports", f"report_{ts}.json")
            mvp._save_json(report, json_path)

            print("📄 위기 분석 MD 리포트 생성 중... (저작권 섹션 제외)")
            md_path = CrisisReportEngineV193(include_copyright=False).create_report(report)

            mvp.print_report(report)
            print(f"\n✅ JSON 보고서 : {json_path}")
            if md_path: print(f"✅ MD  리포트  : {md_path}")
            print("\nℹ️  텍스트 입력은 영상이 없어 저작권 침해 분석을 수행하지 않습니다.")


if __name__ == "__main__":
    main()
