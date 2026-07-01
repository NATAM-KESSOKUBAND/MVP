"""
pdf_report_generator.py — NATAM 스타일 PDF 리포트 생성기
=========================================================
이미지 디자인 기준 3페이지 구성:
  Page 1: 콘텐츠 리스크 분석 요약 (메타 + 핵심요약 + 리스크 레벨)
  Page 2: 주요 리스크 분석 (NATAM A축 · B축 항목별 단계)
  Page 3: 권장 수정 및 대응 가이드 (대응 액션 + 유사 사례)

사용법:
  from pdf_report_generator import generate_pdf_report
  pdf_path = generate_pdf_report(report, output_dir="reports")
"""

import os
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.flowables import Flowable


# ════════════════════════════════════════════════════════
# 🎨 디자인 토큰
# ════════════════════════════════════════════════════════

W, H = A4  # 595.28 x 841.89 pt
PAD  = 20 * mm

# 컬러 팔레트 (이미지 디자인 기준)
C = {
    "bg":          colors.HexColor("#F7F7F5"),   # 페이지 배경 (아이보리)
    "card":        colors.white,
    "card_border": colors.HexColor("#E8E8E4"),
    "primary":     colors.HexColor("#1A1A1A"),   # 메인 텍스트
    "secondary":   colors.HexColor("#6B6B6B"),   # 서브 텍스트
    "accent":      colors.HexColor("#E85D26"),   # 오렌지 포인트
    "blue":        colors.HexColor("#2563EB"),
    "green":       colors.HexColor("#16A34A"),
    "yellow":      colors.HexColor("#D97706"),

    # 리스크 단계별
    "SAFE":     colors.HexColor("#16A34A"),
    "CARE":     colors.HexColor("#2563EB"),
    "ALERT":    colors.HexColor("#D97706"),
    "DANGER":   colors.HexColor("#EA580C"),
    "CRITICAL": colors.HexColor("#DC2626"),

    # 배지 배경
    "SAFE_bg":     colors.HexColor("#DCFCE7"),
    "CARE_bg":     colors.HexColor("#DBEAFE"),
    "ALERT_bg":    colors.HexColor("#FEF3C7"),
    "DANGER_bg":   colors.HexColor("#FFEDD5"),
    "CRITICAL_bg": colors.HexColor("#FEE2E2"),

    "divider": colors.HexColor("#E5E5E0"),
    "header_bg": colors.HexColor("#1A1A1A"),
}

LEVEL_LABEL = {
    "SAFE": "낮음", "CARE": "주의", "ALERT": "중간",
    "DANGER": "높음", "CRITICAL": "심각",
}
LEVEL_ORDER = ["SAFE", "CARE", "ALERT", "DANGER", "CRITICAL"]


# ════════════════════════════════════════════════════════
# 🔤 폰트 등록
# ════════════════════════════════════════════════════════

# 한국어 폰트 후보 경로 (Windows 맑은 고딕 우선, Linux Nanum 대체)
# 한국어 폰트 후보: (regular_path, bold_path, ttc_index_or_None)
_FONT_CANDIDATES = [
    # Windows 맑은 고딕
    ("C:/Windows/Fonts/malgun.ttf",   "C:/Windows/Fonts/malgunbd.ttf",  None),
    # Linux Nanum
    ("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
     "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",         None),
    # macOS Apple SD Gothic (TTC, subfont index 0)
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
     "/System/Library/Fonts/AppleSDGothicNeo.ttc",                       0),
]

def _register_fonts():
    from reportlab.pdfbase.ttfonts import TTFont
    for reg_path, bold_path, ttc_idx in _FONT_CANDIDATES:
        if not (os.path.exists(reg_path) and os.path.exists(bold_path)):
            continue
        try:
            if ttc_idx is not None:
                pdfmetrics.registerFont(TTFont("KorR", reg_path,  subfontIndex=ttc_idx))
                pdfmetrics.registerFont(TTFont("KorB", bold_path, subfontIndex=ttc_idx))
            else:
                pdfmetrics.registerFont(TTFont("KorR", reg_path))
                pdfmetrics.registerFont(TTFont("KorB", bold_path))
            return "KorR", "KorB"
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"

FONT_R, FONT_B = _register_fonts()


# ════════════════════════════════════════════════════════
# 🧩 커스텀 Flowable — 헤더 배너
# ════════════════════════════════════════════════════════

class HeaderBanner(Flowable):
    """각 페이지 상단 NATAM 브랜드 헤더"""
    def __init__(self, page_num: int, total: int = 3, subtitle: str = "콘텐츠 업로드 전 리스크 체크"):
        super().__init__()
        self.page_num = page_num
        self.total    = total
        self.subtitle = subtitle
        self.width    = W - 2 * PAD
        self.height   = 14 * mm

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # 배경
        c.setFillColor(C["header_bg"])
        c.roundRect(0, 0, w, h, 4, fill=1, stroke=0)

        # NATAM 로고
        c.setFillColor(colors.white)
        c.setFont(FONT_B, 11)
        c.drawString(10, h / 2 - 4, "NATAM")

        # 서브타이틀
        c.setFont(FONT_R, 8)
        c.setFillColor(colors.HexColor("#AAAAAA"))
        c.drawString(10, h / 2 - 14, self.subtitle)

        # 페이지 번호
        pg_text = f"{self.page_num}/{self.total}"
        c.setFont(FONT_R, 9)
        c.setFillColor(colors.white)
        c.drawRightString(w - 10, h / 2 - 4, pg_text)


class SectionTitle(Flowable):
    """아이콘 + 대제목 + 부제목 조합 섹션 타이틀"""
    def __init__(self, icon: str, title: str, subtitle: str, icon_color=None):
        super().__init__()
        self.icon       = icon
        self.title      = title
        self.subtitle   = subtitle
        self.icon_color = icon_color or C["accent"]
        self.width      = W - 2 * PAD
        self.height     = 28 * mm

    def draw(self):
        c = self.canv
        h = self.height

        # 아이콘 원
        c.setFillColor(self.icon_color)
        c.circle(15, h - 16, 10, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(FONT_B, 13)
        c.drawCentredString(15, h - 21, self.icon)

        # 제목
        c.setFillColor(C["primary"])
        c.setFont(FONT_B, 18)
        c.drawString(32, h - 14, self.title)

        # 부제목
        c.setFont(FONT_R, 9)
        c.setFillColor(C["secondary"])
        c.drawString(32, h - 26, self.subtitle)

        # 구분선
        c.setStrokeColor(C["divider"])
        c.setLineWidth(0.5)
        c.line(0, 0, self.width, 0)


class RiskBadge(Flowable):
    """단계 배지 (SAFE / CARE / ALERT / DANGER / CRITICAL)"""
    def __init__(self, level: str, width: float = 45, height: float = 14):
        super().__init__()
        self.level  = level
        self.width  = width
        self.height = height

    def draw(self):
        c   = self.canv
        lv  = self.level
        bg  = C.get(f"{lv}_bg",  colors.HexColor("#F3F4F6"))
        fg  = C.get(lv,          colors.HexColor("#6B7280"))
        txt = LEVEL_LABEL.get(lv, lv)

        c.setFillColor(bg)
        c.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=0)
        c.setFillColor(fg)
        c.setFont(FONT_B, 8)
        c.drawCentredString(self.width / 2, 3.5, txt)


# ════════════════════════════════════════════════════════
# 🛠️  스타일 팩토리
# ════════════════════════════════════════════════════════

def _ps(name, font=None, size=9, color=None, leading=None, space_before=0, space_after=2, **kw):
    return ParagraphStyle(
        name,
        fontName     = font or FONT_R,
        fontSize     = size,
        textColor    = color or C["primary"],
        leading      = leading or (size * 1.45),
        spaceBefore  = space_before,
        spaceAfter   = space_after,
        **kw,
    )

ST = {
    "body":       _ps("body"),
    "body_sm":    _ps("body_sm",  size=8,  color=C["secondary"]),
    "bold":       _ps("bold",     font=FONT_B, size=9),
    "bold_sm":    _ps("bold_sm",  font=FONT_B, size=8),
    "label":      _ps("label",    font=FONT_B, size=7,  color=C["secondary"]),
    "card_title": _ps("card_t",   font=FONT_B, size=11),
    "bullet":     _ps("bullet",   size=8,  leftIndent=10, bulletIndent=0),
}


# ════════════════════════════════════════════════════════
# 🃏 카드 빌더
# ════════════════════════════════════════════════════════

def _card_table(content_rows, col_widths=None):
    """흰색 라운드 카드처럼 보이는 단일 셀 테이블"""
    col_w = col_widths or [W - 2 * PAD]
    t = Table([[content_rows]], colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), C["card"]),
        ("ROUNDEDCORNERS", [6]),
        ("BOX",         (0, 0), (-1, -1), 0.5, C["card_border"]),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1),  10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",(0, 0), (-1, -1), 12),
    ]))
    return t


def _two_col_table(left_items, right_items, left_w=None, right_w=None):
    """2컬럼 나란히 테이블"""
    avail = W - 2 * PAD - 6
    lw = left_w  or avail * 0.5
    rw = right_w or avail * 0.5
    t = Table([[left_items, right_items]], colWidths=[lw, rw])
    t.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("INNERGRID",    (0, 0), (-1, -1), 0, colors.transparent),
        ("BOX",          (0, 0), (-1, -1), 0, colors.transparent),
    ]))
    return t


# ════════════════════════════════════════════════════════
# 📊 헬퍼 — 데이터 추출
# ════════════════════════════════════════════════════════

def _safe(val, default="—"):
    if val in (None, "", [], {}): return default
    return str(val).strip() or default

def _overall_level(natam: dict) -> str:
    """A·B축 전체에서 가장 높은 단계 반환"""
    all_levels = []
    for axis in ("A", "B"):
        for v in natam.get(axis, {}).values():
            lv = v.get("level", "SAFE")
            if lv in LEVEL_ORDER:
                all_levels.append(LEVEL_ORDER.index(lv))
    if not all_levels:
        return "SAFE"
    return LEVEL_ORDER[max(all_levels)]

def _overall_label(natam: dict) -> str:
    lv = _overall_level(natam)
    return LEVEL_LABEL.get(lv, lv)

def _count_alerts(natam: dict) -> int:
    count = 0
    for axis in ("A", "B"):
        for v in natam.get(axis, {}).values():
            if v.get("level", "SAFE") not in ("SAFE", "CARE"):
                count += 1
    return count

NATAM_A_NAMES = {
    "A-01": "정치·이념 리스크",        "A-02": "특정 인물·집단 낙인 리스크",
    "A-03": "사생활·윤리 리스크",      "A-04": "이미지 역반전 리스크",
    "A-05": "맥락 절단·클립화 리스크", "A-06": "갈등 증폭 리스크",
    "A-07": "밈화·조롱 소비 리스크",   "A-08": "브랜드 세이프티 리스크",
    "A-09": "감정 선동 리스크",        "A-10": "기존 논란 결합 리스크",
}
NATAM_B_NAMES = {
    "B-01": "괴롭힘·모욕 표현 리스크",  "B-02": "폭력·위협·불법행위 리스크",
    "B-03": "혐오·차별 표현 리스크",    "B-04": "성적 표현·대상화 리스크",
    "B-05": "광고친화성·상업 신뢰 리스크",
}

ICON_BY_NAME = {
    "정치·이념 리스크":         "⚑",
    "특정 인물·집단 낙인 리스크": "👤",
    "사생활·윤리 리스크":       "🔒",
    "이미지 역반전 리스크":     "↩",
    "맥락 절단·클립화 리스크":  "✂",
    "갈등 증폭 리스크":         "⚡",
    "밈화·조롱 소비 리스크":    "😂",
    "브랜드 세이프티 리스크":   "🏷",
    "감정 선동 리스크":         "🔥",
    "기존 논란 결합 리스크":    "🔗",
    "괴롭힘·모욕 표현 리스크":  "⚠",
    "폭력·위협·불법행위 리스크": "🚨",
    "혐오·차별 표현 리스크":    "🚫",
    "성적 표현·대상화 리스크":  "🔞",
    "광고친화성·상업 신뢰 리스크": "💰",
}


# ════════════════════════════════════════════════════════
# 📄 Page 1 — 콘텐츠 리스크 분석 요약
# ════════════════════════════════════════════════════════

def _build_page1(report: dict) -> list:
    meta   = report.get("meta", {})
    natam  = report.get("natam_risk", {})
    cls    = report.get("classification", {})
    spread = report.get("spread_stage", {})
    yt     = report.get("youtube_meta", {})

    overall_lv    = _overall_level(natam)
    overall_label = _overall_label(natam)
    alert_count   = _count_alerts(natam)

    story = []

    # ── 헤더 ──
    story.append(HeaderBanner(1))
    story.append(Spacer(1, 6 * mm))

    # ── 섹션 타이틀 ──
    story.append(SectionTitle("🛡", "콘텐츠 리스크 분석 요약", "업로드 전 잠재 리스크를 미리 확인하세요"))
    story.append(Spacer(1, 5 * mm))

    # ── 분석 대상 영상 카드 ──
    title     = _safe(meta.get("incident_title") or meta.get("video_filename") or yt.get("title"), "영상 제목 없음")
    channel   = _safe(yt.get("channel") or meta.get("youtube_channel"), "로컬 파일")
    platform  = "YouTube" if yt.get("url") else "로컬 파일"
    dur       = _safe(meta.get("duration_sec"), "—")
    analyzed  = _safe(meta.get("analyzed_at"), datetime.now().strftime("%Y-%m-%d %H:%M"))
    upload_dt = _safe(yt.get("upload_date"), "—")
    if upload_dt != "—" and len(upload_dt) == 8:
        upload_dt = f"{upload_dt[:4]}-{upload_dt[4:6]}-{upload_dt[6:]}"

    # 영상 정보 2컬럼 그리드
    info_data = [
        [Paragraph("분석 대상 영상", ST["label"]),  ""],
        [Paragraph(f"<b>{title[:45]}{'…' if len(title)>45 else ''}</b>",
                   _ps("t2", font=FONT_B, size=10)), ""],
        [Paragraph(f"{channel}", ST["body_sm"]),    ""],
        ["", ""],
        [Paragraph("📺  업로드 플랫폼", ST["body_sm"]),
         Paragraph(f"📅  업로드 예정일 {upload_dt}", ST["body_sm"])],
        [Paragraph(f"<font color='#E85D26'><b>{platform}</b></font>",
                   _ps("pl", font=FONT_B, size=9, color=C["accent"])),
         Paragraph(f"⏱  분석 일시 {analyzed[:16]}", ST["body_sm"])],
    ]
    info_t = Table(info_data, colWidths=[(W - 2*PAD - 24) * 0.6, (W - 2*PAD - 24) * 0.4])
    info_t.setStyle(TableStyle([
        ("SPAN",         (0, 0), (1, 0)),
        ("SPAN",         (0, 1), (1, 1)),
        ("SPAN",         (0, 2), (1, 2)),
        ("SPAN",         (0, 3), (1, 3)),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW",    (0, 3), (1, 3), 0.5, C["divider"]),
    ]))

    card_content = [info_t]
    story.append(_wrap_in_card(card_content))
    story.append(Spacer(1, 4 * mm))

    # ── 핵심 요약 카드 ──
    summary_text = _safe(
        meta.get("incident_summary") or meta.get("input_query"),
        "자막 분석 기반으로 생성된 위기 관리 분석 보고서입니다."
    )
    if len(summary_text) > 200:
        summary_text = summary_text[:200] + "…"

    summary_items = [
        Paragraph("핵심 요약", ST["label"]),
        Spacer(1, 3),
        Paragraph(summary_text, ST["body"]),
        Spacer(1, 5 * mm),
        # 4개 지표 박스
        _build_metric_row(overall_label, overall_lv, alert_count, spread.get("stage","—")),
    ]
    story.append(_wrap_in_card(summary_items))
    story.append(Spacer(1, 4 * mm))

    # ── 리스크 수준 안내 카드 ──
    guide_items = [
        Paragraph("리스크 수준 안내", ST["label"]),
        Spacer(1, 4),
        _build_level_guide(),
    ]
    story.append(_wrap_in_card(guide_items))

    return story


def _wrap_in_card(items: list) -> Table:
    """items 리스트를 단일 셀 흰색 카드로 감싸기"""
    inner = Table([[items]], colWidths=[W - 2 * PAD - 2])
    inner.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), C["card"]),
        ("BOX",          (0, 0), (-1, -1), 0.5, C["card_border"]),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    return inner


def _build_metric_row(overall_label: str, overall_lv: str, alert_count: int, stage: str) -> Table:
    """4개 지표 나란히"""
    def _metric_cell(label, value, value_color=None):
        vc = value_color or C["primary"]
        return [
            Paragraph(label, _ps("ml", size=7, color=C["secondary"])),
            Paragraph(f"<b>{value}</b>", _ps("mv", font=FONT_B, size=14, color=vc)),
        ]

    lv_color = C.get(overall_lv, C["primary"])

    cells = [
        _metric_cell("종합 리스크 수준", overall_label, lv_color),
        _metric_cell("예상 영향도", "보통"),
        _metric_cell("권장 조치 항목", f"{alert_count}건"),
        _metric_cell("확산 단계", stage),
    ]

    row_data  = [[Table([[c]], colWidths=[(W - 2*PAD - 28) / 4]) for c in cells]]
    t = Table(row_data, colWidths=[(W - 2*PAD - 28) / 4] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#F9F9F7")),
        ("BOX",          (0, 0), (-1, -1), 0.5, C["divider"]),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, C["divider"]),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _build_level_guide() -> Table:
    rows = [
        (C["SAFE"],    "🟢", "낮음", "플랫폼 정책 위반 가능성이 낮고, 이슈 발생 가능성이 낮은 상태"),
        (C["ALERT"],   "🟡", "중간", "일부 주의가 필요한 요소가 있으며, 수정/보완을 권장하는 상태"),
        (C["CRITICAL"],"🔴", "높음", "정책 위반 또는 논란 발생 가능성이 높아, 수정 및 주의가 필요한 상태"),
    ]
    data = []
    for color, emoji, label, desc in rows:
        data.append([
            Paragraph(f"<font color='{color.hexval()}'>{emoji}</font>",
                      _ps("gi", size=10)),
            Paragraph(f"<b>{label}</b>: {desc}",
                      _ps("gd", size=8, color=C["secondary"])),
        ])
    t = Table(data, colWidths=[14, W - 2*PAD - 28 - 14])
    t.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


# ════════════════════════════════════════════════════════
# 📄 Page 2 — 주요 리스크 분석
# ════════════════════════════════════════════════════════

def _build_page2(report: dict) -> list:
    natam = report.get("natam_risk", {})
    story = [PageBreak()]

    story.append(HeaderBanner(2))
    story.append(Spacer(1, 6 * mm))
    story.append(SectionTitle("⚠", "주요 리스크 분석",
                              "5가지 핵심 영역별 리스크 수준과 세부 내용을 확인하세요.",
                              icon_color=C["ALERT"]))
    story.append(Spacer(1, 5 * mm))

    # ALERT 이상 항목을 앞으로, 나머지는 뒤에 배치
    all_items = []
    for key, name in NATAM_A_NAMES.items():
        item = natam.get("A", {}).get(key, {})
        all_items.append((key, name, item.get("level","SAFE"), item.get("reason","—")))
    for key, name in NATAM_B_NAMES.items():
        item = natam.get("B", {}).get(key, {})
        all_items.append((key, name, item.get("level","SAFE"), item.get("reason","—")))

    # ALERT 이상 먼저, 그 다음 SAFE/CARE
    priority   = [i for i in all_items if i[2] not in ("SAFE", "CARE")]
    rest       = [i for i in all_items if i[2] in ("SAFE", "CARE")]
    show_items = priority[:6]  # ALERT 이상 항목만, 최대 6개 (SAFE/CARE 제외로 중요도 집중)

    for key, name, level, reason in show_items:
        story.append(_build_risk_item_card(key, name, level, reason))
        story.append(Spacer(1, 3 * mm))

    return story


def _build_risk_item_card(key: str, name: str, level: str, reason: str) -> Table:
    lv_color = C.get(level, C["secondary"])
    bg_color = C.get(f"{level}_bg", colors.HexColor("#F9F9F7"))
    label    = LEVEL_LABEL.get(level, level)
    icon     = ICON_BY_NAME.get(name, "•")

    # 배지
    badge_data = [[
        Paragraph(f"<font color='{lv_color.hexval()}'><b>{label}</b></font>",
                  _ps("bdg", font=FONT_B, size=8, color=lv_color)),
    ]]
    badge_t = Table(badge_data, colWidths=[30])
    badge_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg_color),
        ("ROUNDEDCORNERS",[5]),
        ("TOPPADDING",   (0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("RIGHTPADDING", (0,0),(-1,-1), 5),
    ]))

    # 제목 행
    header_row = Table([[
        Paragraph(f"<b>{name}</b>", _ps("rn", font=FONT_B, size=10)),
        badge_t,
    ]], colWidths=[W - 2*PAD - 28 - 40, 40])
    header_row.setStyle(TableStyle([
        ("VALIGN",      (0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING", (0,0),(-1,-1), 0),
        ("RIGHTPADDING",(0,0),(-1,-1), 0),
        ("TOPPADDING",  (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))

    # 이유 텍스트
    reason_short = reason[:120] + "…" if len(reason) > 120 else reason

    # 주요 포인트 (reason에서 첫 문장 추출)
    first_sentence = reason_short.split(".")[0].strip()

    card_inner = [
        header_row,
        Spacer(1, 4),
        Paragraph(reason_short, ST["body_sm"]),
        Spacer(1, 4),
        Paragraph("주요 포인트", ST["label"]),
        Paragraph(f"• {first_sentence}", ST["bullet"]),
    ]

    return _wrap_in_card(card_inner)


# ════════════════════════════════════════════════════════
# 📄 Page 3 — 권장 수정 및 대응 가이드
# ════════════════════════════════════════════════════════

def _build_page3(report: dict) -> list:
    meta   = report.get("meta", {})
    spread = report.get("spread_stage", {})
    cases  = report.get("similar_cases", [])
    wa     = report.get("worst_actions", [])

    stage = spread.get("stage", "Early")
    actions = {
        "Early": {
            "immediate": ["문제 구간 내부 검토 및 증거 보전", "커뮤니티·SNS 언급량 모니터링 시작"],
            "short":     ["법률·PR 전문가와 대응 방향 사전 협의", "공식 입장문 초안 작성"],
        },
        "Mid": {
            "immediate": ["문제 콘텐츠 접근 제한 또는 수정본 업로드 검토", "공식 입장문 즉시 작성 및 채널 공지"],
            "short":     ["커뮤니티 댓글·DM 모니터링 및 대응 방침 수립", "미디어 문의 대응 창구 일원화"],
        },
        "Late": {
            "immediate": ["추가 악화 요인 차단 (불필요한 발언·라이브 자제)", "기존 공개 입장문 일관성 유지"],
            "short":     ["장기적 이미지 회복 전략 수립", "법적 리스크 최종 점검"],
        },
    }.get(stage, {
        "immediate": ["확산 지표 데이터 수집 우선 진행", "내부 상황 파악 및 관계자 보고"],
        "short":     ["전문가 자문 의뢰", "모니터링 체계 구축"],
    })

    story = [PageBreak()]

    story.append(HeaderBanner(3))
    story.append(Spacer(1, 6 * mm))
    story.append(SectionTitle("✅", "권장 수정 및 대응 가이드",
                              "아래 권장 사항을 반영하면 리스크를 줄이고 콘텐츠 안정성을 높일 수 있습니다.",
                              icon_color=C["green"]))
    story.append(Spacer(1, 5 * mm))

    # ── 대응 가이드 섹션 (action 번호별) ──
    natam  = report.get("natam_risk", {})
    all_items = []
    for key, name in NATAM_A_NAMES.items():
        item = natam.get("A", {}).get(key, {})
        all_items.append((name, item.get("level","SAFE"), item.get("reason","—")))
    for key, name in NATAM_B_NAMES.items():
        item = natam.get("B", {}).get(key, {})
        all_items.append((name, item.get("level","SAFE"), item.get("reason","—")))

    alert_items = [i for i in all_items if i[1] not in ("SAFE", "CARE")][:4]

    for idx, (name, level, reason) in enumerate(alert_items, 1):
        lv_color = C.get(level, C["secondary"])
        label    = LEVEL_LABEL.get(level, level)

        badge = Table([[
            Paragraph(f"<b>{label}</b>", _ps("ab", font=FONT_B, size=7, color=lv_color)),
        ]], colWidths=[28])
        badge.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), C.get(f"{level}_bg", colors.white)),
            ("TOPPADDING",   (0,0),(-1,-1), 2),
            ("BOTTOMPADDING",(0,0),(-1,-1), 2),
            ("LEFTPADDING",  (0,0),(-1,-1), 4),
            ("RIGHTPADDING", (0,0),(-1,-1), 4),
        ]))

        title_row = Table([[
            Paragraph(f"<b>{idx}. {name}</b>", _ps("an", font=FONT_B, size=10)),
            badge,
        ]], colWidths=[W - 2*PAD - 28 - 38, 38])
        title_row.setStyle(TableStyle([
            ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
            ("TOPPADDING",   (0,0),(-1,-1), 0),
            ("BOTTOMPADDING",(0,0),(-1,-1), 0),
        ]))

        # 권장 bullet
        rec_text = _get_recommendation(name, reason)
        bullets  = [Paragraph(f"• {b}", ST["bullet"]) for b in rec_text[:2]]

        card_inner = [title_row, Spacer(1, 4)] + bullets
        story.append(_wrap_in_card(card_inner))
        story.append(Spacer(1, 3 * mm))

    # 권장 항목이 부족하면 즉시 대응 행동 카드 추가
    if len(alert_items) < 2:
        imm_bullets = [Paragraph(f"• {a}", ST["bullet"]) for a in actions["immediate"]]
        sht_bullets = [Paragraph(f"• {a}", ST["bullet"]) for a in actions["short"]]
        card_inner = [
            Paragraph("즉시 대응 (24시간 이내)", ST["bold"]),
            Spacer(1, 4),
        ] + imm_bullets + [
            Spacer(1, 4),
            Paragraph("단기 대응 (3일 이내)", ST["bold"]),
            Spacer(1, 4),
        ] + sht_bullets
        story.append(_wrap_in_card(card_inner))
        story.append(Spacer(1, 3 * mm))

    # ── 추가 참고 사항 ──
    note_inner = [
        Paragraph("추가 참고 사항", ST["bold"]),
        Spacer(1, 4),
        Paragraph("• 위 분석은 업로드 전 사전 검토 기준이며, 플랫폼 정책 변경에 따라 리스크 수준이 달라질 수 있습니다.",
                  ST["body_sm"]),
        Paragraph("• 중요한 이슈가 발생할 경우 즉시 콘텐츠 수정 또는 비공개 전환을 권장합니다.",
                  ST["body_sm"]),
    ]
    story.append(_wrap_in_card(note_inner))
    story.append(Spacer(1, 4 * mm))

    # ── CTA 배너 ──
    story.append(_build_cta_banner())

    return story


def _get_recommendation(name: str, reason: str) -> list:
    """항목명 + reason으로 권장 수정 사항 생성 (간단 규칙 기반)"""
    recs = {
        "정치·이념 리스크":         ["정치적 단정 표현을 완화하거나 복수 시각을 제시해 주세요.",
                                     "진영 프레이밍 표현은 삭제하거나 중립 표현으로 대체해 주세요."],
        "특정 인물·집단 낙인 리스크": ["실명 언급 시 사실 확인된 내용만 포함해 주세요.",
                                     "부정적 프레이밍의 반복 표현을 제거해 주세요."],
        "사생활·윤리 리스크":        ["개인 정보·가족사 관련 언급은 당사자 동의 여부를 확인해 주세요.",
                                     "사적 정보 공개가 불필요한 부분은 편집해 주세요."],
        "이미지 역반전 리스크":       ["기존 콘텐츠 방향성과 상충되는 표현을 재검토해 주세요.",
                                     "위선 프레임으로 소비될 수 있는 장면을 삭제하거나 맥락 설명을 추가해 주세요."],
        "맥락 절단·클립화 리스크":   ["자극적인 문구나 과장된 표현은 오해를 유발할 수 있습니다.",
                                     "실제 내용과 일치하는 썸네일/제목으로 신뢰도를 높여주세요."],
        "갈등 증폭 리스크":           ["편가르기 구조의 표현을 중립적으로 수정해 주세요.",
                                     "논쟁 유도 표현 대신 정보 전달 중심으로 구성해 주세요."],
        "밈화·조롱 소비 리스크":      ["어색한 표현이나 극단적 리액션은 편집하거나 자막으로 보완해 주세요.",
                                     "의도치 않게 클립 소비될 수 있는 구간을 재검토해 주세요."],
        "브랜드 세이프티 리스크":     ["광고주 비친화적 표현(과도한 욕설, 자극적 분위기)을 완화해 주세요.",
                                     "브랜드 이미지와 충돌하는 요소를 제거하거나 수위를 조절해 주세요."],
        "감정 선동 리스크":           ["분노 유도 구조의 표현을 완화하거나 팩트 중심으로 재구성해 주세요.",
                                     "과격한 단정 표현 대신 '~한 경향이 있다' 식의 완화 표현을 사용해 주세요."],
        "기존 논란 결합 리스크":      ["과거 논란과 연결될 수 있는 소재나 표현을 삭제해 주세요.",
                                     "'원래 저랬다' 소비 구조를 방지하기 위해 맥락 설명을 추가해 주세요."],
        "괴롭힘·모욕 표현 리스크":    ["인신공격·조롱 표현을 삭제하거나 중립 표현으로 교체해 주세요.",
                                     "지속적 비하 발언이 포함된 구간은 편집해 주세요."],
        "폭력·위협·불법행위 리스크":  ["폭력 묘사나 위협 발언이 포함된 구간을 삭제해 주세요.",
                                     "위험행동을 조장하는 표현은 반드시 편집하거나 경고 문구를 추가해 주세요."],
        "혐오·차별 표현 리스크":      ["성별·인종·지역 일반화 표현을 삭제하거나 수정해 주세요.",
                                     "혐오 밈이나 차별 표현은 플랫폼 정책 위반이 될 수 있습니다."],
        "성적 표현·대상화 리스크":    ["성적 암시나 신체 대상화 표현은 삭제하거나 수위를 낮춰 주세요.",
                                     "선정성이 높은 구간은 편집하거나 연령 제한 설정을 검토해 주세요."],
        "광고친화성·상업 신뢰 리스크":["광고 협찬 표기 여부를 반드시 확인해 주세요.",
                                     "과도한 욕설이나 브랜드 세이프티 충돌 요소를 완화해 주세요."],
    }
    return recs.get(name, [
        "해당 리스크 요소를 재검토하고 필요 시 수정해 주세요.",
        reason[:60] + ("…" if len(reason) > 60 else ""),
    ])


def _build_cta_banner() -> Table:
    inner = [
        Paragraph("<b>더 안전한 콘텐츠 운영을 위한</b>", _ps("ct1", font=FONT_B, size=12, color=C["primary"])),
        Paragraph("<b>리스크 체크가 필요하신가요?</b>", _ps("ct2", font=FONT_B, size=12, color=C["primary"])),
        Spacer(1, 4),
        Paragraph("NATAM은 업로드 전 잠재 리스크를 미리 분석하여",  ST["body_sm"]),
        Paragraph("콘텐츠가 더 안전하게 사랑받을 수 있도록 도와드립니다.", ST["body_sm"]),
        Spacer(1, 4),
        Paragraph("natam.ai", _ps("url", font=FONT_B, size=9, color=C["blue"])),
    ]
    t = Table([[inner]], colWidths=[W - 2*PAD - 2])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor("#F0F4FF")),
        ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#BFCFEF")),
        ("TOPPADDING",   (0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING",  (0,0),(-1,-1), 14),
        ("RIGHTPADDING", (0,0),(-1,-1), 14),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    return t


# ════════════════════════════════════════════════════════
# 🦶 푸터 콜백
# ════════════════════════════════════════════════════════

def _footer_callback(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_R, 7)
    canvas.setFillColor(C["secondary"])
    footer_text = "본 분석은 자동화와 전문 리뷰를 기반으로 제공되며, 최종 판단과 책임은 콘텐츠 운영자에게 있습니다."
    canvas.drawCentredString(W / 2, 12 * mm, footer_text)
    canvas.restoreState()


# ════════════════════════════════════════════════════════
# 🏗️  메인 빌더
# ════════════════════════════════════════════════════════

def generate_pdf_report(report: dict, output_dir: str = "reports") -> str:
    """
    report dict → PDF 파일 생성

    Parameters
    ----------
    report     : analyze_video_full() 가 반환하는 report dict
    output_dir : 저장 폴더

    Returns
    -------
    str : 생성된 PDF 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)

    meta     = report.get("meta", {})
    base     = (meta.get("video_filename") or "report").replace(".", "_").replace(" ", "_")
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"NATAM_REPORT_{base}_{ts}.pdf"
    path     = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(
        path,
        pagesize    = A4,
        leftMargin  = PAD,
        rightMargin = PAD,
        topMargin   = PAD,
        bottomMargin= 18 * mm,
    )

    story = []
    story += _build_page1(report)
    story += _build_page2(report)
    story += _build_page3(report)

    doc.build(story, onFirstPage=_footer_callback, onLaterPages=_footer_callback)
    print(f"✅ PDF 리포트 생성 완료 → {path}")
    return path


# ════════════════════════════════════════════════════════
# 🧪 독립 실행 테스트
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    dummy_report = {
        "meta": {
            "video_filename":  "00치킨_신메뉴_솔직_리뷰.mp4",
            "analyzed_at":     "2024-06-13 14:32:00",
            "duration_sec":    843,
            "total_segments":  72,
            "total_corrected": 8,
            "incident_title":  "00치킨 신메뉴 솔직 리뷰 영상 리스크 분석",
            "incident_summary": (
                "전반적으로 심각한 리스크는 낮지 않으나, 일부 표현과 맥락에서 오해 소지가 "
                "발생할 수 있는 요소가 확인되었습니다. 아래 주요 리스크 항목을 확인하고 "
                "권장 사항을 반영하시면 콘텐츠 안정성을 높일 수 있습니다."
            ),
        },
        "youtube_meta": {
            "title":            "[HD/60p] 00치킨 신메뉴 솔직 리뷰!",
            "channel":          "솔직리뷰TV",
            "url":              "https://youtube.com/watch?v=EXAMPLE",
            "view_count":       48200,
            "like_count":       1830,
            "comment_count":    312,
            "subscriber_count": 182000,
            "upload_date":      "20240615",
        },
        "spread_stage": {"stage": "Early", "reasons": ["초기 단계"], "metrics": None},
        "classification": {"labels": ["L05 기만 행위"], "primary": "L05", "reason": "광고 미표기 의심"},
        "similar_cases": [
            {"rank":1,"title":"대형 유튜버 뒷광고 대란","controversy_type":"advertising_issue","distance":0.312,"response_pattern":["초반 부인","증거 폭로 후 사과"]},
        ],
        "worst_actions": ["감정적으로 즉각 반박하는 영상 업로드"],
        "transcript_analysis": [],
        "keyframe_analysis":   [],
        "transcript_files":    {},
        "natam_risk": {
            "A": {
                "A-01": {"level": "SAFE",   "reason": "정치적 요소가 감지되지 않음"},
                "A-02": {"level": "CARE",   "reason": "특정 브랜드 비교 표현이 반복됨"},
                "A-03": {"level": "SAFE",   "reason": "사생활 관련 언급 없음"},
                "A-04": {"level": "ALERT",  "reason": "선행 이미지와 상충하는 상업적 표현 존재"},
                "A-05": {"level": "ALERT",  "reason": "자극적 문구가 클립으로 소비될 가능성"},
                "A-06": {"level": "CARE",   "reason": "경쟁 브랜드 언급 시 불필요한 논쟁 발생 가능성"},
                "A-07": {"level": "CARE",   "reason": "과장 표현이 밈으로 소비될 소지"},
                "A-08": {"level": "ALERT",  "reason": "협찬 표기 부재로 광고주 비친화 리스크"},
                "A-09": {"level": "SAFE",   "reason": "감정 선동 요소 감지되지 않음"},
                "A-10": {"level": "SAFE",   "reason": "기존 논란 결합 요소 없음"},
            },
            "B": {
                "B-01": {"level": "SAFE",   "reason": "모욕 표현 없음"},
                "B-02": {"level": "SAFE",   "reason": "폭력·위협 표현 없음"},
                "B-03": {"level": "SAFE",   "reason": "혐오·차별 표현 없음"},
                "B-04": {"level": "SAFE",   "reason": "성적 표현 없음"},
                "B-05": {"level": "ALERT",  "reason": "협찬 표기가 없어 광고 정책 위반 가능성 있음"},
            },
            "overall_a": "ALERT",
            "overall_b": "ALERT",
        },
        "rule_scan": {"hit": False},
        "pattern_summary": "",
    }

    path = generate_pdf_report(dummy_report, output_dir="/mnt/user-data/outputs")
    print(f"저장: {path}")