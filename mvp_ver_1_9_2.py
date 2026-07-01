import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import whisper
from google import genai
import time
import threading
import torch
import json
import yaml
import numpy as np
import faiss
import re
import cv2
import PIL.Image
import joblib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# PDF 리포트 생성기 (같은 폴더에 pdf_report_generator.py 필요)
try:
    from pdf_report_generator import generate_pdf_report as _gen_pdf
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False


# ════════════════════════════════════════════════════════
# ⚙️  설정
# ════════════════════════════════════════════════════════

API_KEY                  = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL              = os.getenv("EMBED_MODEL", "models/gemini-embedding-2")    # 001 → 2 (최신)
GEN_MODEL                = os.getenv("GEN_MODEL",   "models/gemini-3.5-flash")       # 2.5 → 3.5 (최신 flash)
WHISPER_MODEL            = "turbo"
GEMINI_REFINE_MODEL      = os.getenv("GEMINI_REFINE_MODEL", "models/gemini-3.5-flash")
TRANSCRIBE_ENGINE        = os.getenv("TRANSCRIBE_ENGINE", "gemini")  # "whisper" | "gemini" | "clova"
CONTROVERSY_MODEL_PATH   = "controversy_model.joblib"  # ML 분류 모델
_REFINE_CONCURRENCY      = 4                           # Gemini 교정 배치 동시 실행 수

# ── [1순위] Clova NEST(Speech Recognition) ASR — 한국어 CER 우위 가설 검증용 ──
#   네이버 클라우드 콘솔에서 Clova Speech / NEST 발급 후 env 주입.
#   키가 없으면 자동으로 기존 엔진(gemini)으로 폴백한다.
CLOVA_INVOKE_URL  = os.getenv("CLOVA_INVOKE_URL", "")     # 예: https://clovaspeech-gw.../recognizer/upload
CLOVA_SECRET_KEY  = os.getenv("CLOVA_SECRET_KEY", "")
FFMPEG_BIN        = os.getenv("FFMPEG_BIN", "ffmpeg")     # 영상→오디오 추출용

# ── [2순위] 2-Tier 분류 파라미터 ──
#   Tier1(SVM): 재현율 우선 → 의심 구간을 '넓게' 트리거 (낮은 임계값)
#   Tier2(LLM): 트리거된 구간만 맥락+유사사례+룰매칭과 함께 최종 판정
TIER1_TRIGGER_THRESHOLD = float(os.getenv("TIER1_TRIGGER_THRESHOLD", "0.15"))  # 0.3 → 0.15 (recall↑)
TIER2_CONTEXT_WINDOW    = 1     # 트리거 세그먼트 앞뒤 N개를 맥락으로 동봉
TIER2_MAX_SEGMENTS      = 25    # 비용 상한: Tier2로 보낼 최대 세그먼트 수
ENABLE_TIER2            = os.getenv("ENABLE_TIER2", "1") == "1"

# 고심각도 라벨 — false negative 비용이 큰 카테고리.
#   이 라벨이 Tier1에서 잡히면 '무조건' Tier2 LLM 검토를 강제한다(심각도별 컴퓨트 라우팅).
#   ※ 역사왜곡·고인모독·미성년 등은 현 L01~L12 체계에 없으므로 추후 라벨 확장 시 여기에 추가.
HIGH_SEVERITY_LABELS = set(
    os.getenv("HIGH_SEVERITY_LABELS", "L03,L06,L09,L11").split(",")
)

# ── 비용 계측 토글 (SaaS 가격 산정 근거) ──
ENABLE_COST_METERING = os.getenv("ENABLE_COST_METERING", "1") == "1"
# 1K 토큰당 USD.  ⚠️ 아래 기본값은 구(2.5-flash) 추정치이며 gemini-3.5-flash 실단가가 아님.
#   Google AI 가격표에서 3.5-flash 입력/출력 단가를 확인해 아래 env 로 주입할 것:
#     $env:PRICE_PER_1K_IN="..."; $env:PRICE_PER_1K_OUT="..."
#   (정확한 단가를 넣기 전까지 report.meta.cost.est_usd 는 어림치임)
_PRICE_PER_1K_IN  = float(os.getenv("PRICE_PER_1K_IN",  "0.000075"))
_PRICE_PER_1K_OUT = float(os.getenv("PRICE_PER_1K_OUT", "0.00030"))

LABEL_INFO = (
    "L01 직접적 욕설, L02 인신공격/비하, L03 혐오 표현, L04 허위 정보, "
    "L06 위험/자해 행동, L07 정치적 편향, L08 사생활 침해, "
    "L09 성적 불쾌감, L11 피해자 조롱, L12 해당 없음"
    "  (※ L05 기만·L10 저작권은 별도 Copyright Detector가 담당 — 문장 분류 제외)"
)

ABBREVIATION_MAP = {
    "근데":"그런데","글구":"그리고","아님":"아니면","암튼":"아무튼",
    "젤":"가장","넘":"너무","디게":"되게","진짜루":"진짜로",
    "솔까":"솔직히","이거":"이것","그거":"그것","저거":"저것",
    "이건":"이것은","그건":"그것은","저건":"저것은",
    "이게":"이것이","그게":"그것이","저게":"저것이",
    "어케":"어떻게","왜케":"왜 이렇게","담에":"다음에",
    "맨날":"매일","거임":"것임","거다":"것이다",
    "거에요":"것이에요","됌":"됨","안됌":"안 됨",
    "안됨":"안 됨","아뇨":"아니요","알랴줌":"알려줌","됐음":"되었습니다",
}

YOUTUBE_URL_PATTERNS = re.compile(
    r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+'
)
GOOGLE_DRIVE_URL_PATTERN = re.compile(
    r'https?://drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^&]*&)*id=)([\w\-]+)'
)


# ════════════════════════════════════════════════════════
# 📐 CER (Character Error Rate) — ASR A/B 테스트용 [1순위]
# ════════════════════════════════════════════════════════
def _edit_distance(a: str, b: str) -> int:
    """문자 단위 레벤슈타인 거리 (CER 계산용)."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return max(m, n)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def cer(reference: str, hypothesis: str) -> float:
    """
    한국어는 WER보다 CER이 정확. 0에 가까울수록 좋음.
    reference: 정답 전사(사람이 만든 골든), hypothesis: ASR 출력.
    """
    ref = re.sub(r"\s+", "", reference or "")
    hyp = re.sub(r"\s+", "", hypothesis or "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


# ════════════════════════════════════════════════════════
# 💰 비용/토큰 계측 — 분석 1건당 단가 추적 (SaaS 가격 근거)
# ════════════════════════════════════════════════════════
class CostTracker:
    """genai 응답의 usage_metadata를 누적해 분석 1건당 토큰·USD를 집계."""

    def __init__(self):
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self._lock = threading.Lock()

    def add(self, resp, label: str = ""):
        if not ENABLE_COST_METERING:
            return
        um = getattr(resp, "usage_metadata", None)
        if um is None:
            return
        pin  = getattr(um, "prompt_token_count", 0) or 0
        pout = getattr(um, "candidates_token_count", 0) or 0
        with self._lock:
            self.calls += 1
            self.in_tokens += pin
            self.out_tokens += pout

    def summary(self) -> dict:
        usd = (self.in_tokens / 1000.0) * _PRICE_PER_1K_IN \
            + (self.out_tokens / 1000.0) * _PRICE_PER_1K_OUT
        return {
            "llm_calls":   self.calls,
            "in_tokens":   self.in_tokens,
            "out_tokens":  self.out_tokens,
            "est_usd":     round(usd, 6),
        }


# ════════════════════════════════════════════════════════
# 👁️  시각 모달리티 [4순위 — 해자(moat)]
#   범용 비전 모델이 못 잡는 한국 특화 트리거를 명시적으로 처리.
# ════════════════════════════════════════════════════════
_EASYOCR_READER = None   # 지연 초기화 (무거움)


def _ocr_screen_text(img_path: str) -> str:
    """
    화면 자막/텍스트 OCR (한국어+영어). easyocr 가 있으면 사용, 없으면 빈 문자열.
    음성에는 없지만 화면 자막에만 있는 욕설/혐오 표현을 잡기 위함.
    """
    global _EASYOCR_READER
    try:
        if _EASYOCR_READER is None:
            import easyocr  # 선택적 의존성
            _EASYOCR_READER = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        lines = _EASYOCR_READER.readtext(img_path, detail=0)
        return " ".join(lines).strip()
    except ImportError:
        return ""   # easyocr 미설치 → 조용히 스킵 (pip install easyocr 로 활성화)
    except Exception:
        return ""


def _detect_korea_visual_triggers(img_path: str) -> list:
    """
    한국 특화 시각 트리거(집게손 제스처, 특정 로고/밈) 탐지 — 확장 포인트.

    범용 비전 모델이 절대 따라오지 못하는 '해자' 영역.
    자체 데이터셋으로 파인튜닝한 YOLO 또는 CLIP 임베딩 매칭을 여기에 연결한다.
    현재는 모델 미연동 상태이므로 빈 리스트를 반환(기존 동작 영향 없음).

    연결 예:
        model = _load_gesture_yolo()          # 자체 학습 가중치
        dets  = model(img_path)
        return [{"type":"pinch_gesture","conf":0.91,"bbox":[...]}, ...]
    """
    return []


# ════════════════════════════════════════════════════════
# 🛡️  NATAM Risk Framework v2.0
# ════════════════════════════════════════════════════════

# 단계 순서 (낮을수록 안전)
NATAM_LEVELS = ["SAFE", "CARE", "ALERT", "DANGER", "CRITICAL"]

# A축 — 커뮤니티 리스크 정의
NATAM_A_AXES = {
    "A-01": {
        "label_id": "COMMUNITY_POLITICAL_IDEOLOGY",
        "name":     "정치·이념 리스크",
        "desc":     "정치적 진영·이념·사회 갈등 요소로 인한 커뮤니티 논쟁 확산 가능성",
    },
    "A-02": {
        "label_id": "COMMUNITY_STIGMATIZATION",
        "name":     "특정 인물·집단 낙인 리스크",
        "desc":     "실존 인물 또는 특정 집단에 대한 반복적 부정 이미지 형성 가능성",
    },
    "A-03": {
        "label_id": "COMMUNITY_PRIVACY_ETHICS",
        "name":     "사생활·윤리 리스크",
        "desc":     "사적 정보·가족사·과거 이력 등이 논란 소비 대상으로 사용될 가능성",
    },
    "A-04": {
        "label_id": "COMMUNITY_IMAGE_REVERSAL",
        "name":     "이미지 역반전 리스크",
        "desc":     "기존 이미지와 상반된 행동·발언으로 위선 프레임이 형성될 가능성",
    },
    "A-05": {
        "label_id": "COMMUNITY_CONTEXT_TRUNCATION",
        "name":     "맥락 절단·클립화 리스크",
        "desc":     "발언 일부만 소비되어 원 의도와 다르게 확산될 구조적 위험",
    },
    "A-06": {
        "label_id": "COMMUNITY_CONFLICT_ESCALATION",
        "name":     "갈등 증폭 리스크",
        "desc":     "댓글·커뮤니티 내 집단 갈등이 증폭될 가능성",
    },
    "A-07": {
        "label_id": "COMMUNITY_MEMEFICATION",
        "name":     "밈화·조롱 소비 리스크",
        "desc":     "발언·표정·행동 일부가 밈 또는 조롱 형태로 소비될 가능성",
    },
    "A-08": {
        "label_id": "COMMUNITY_BRAND_SAFETY",
        "name":     "브랜드 세이프티 리스크",
        "desc":     "브랜드·광고주가 연계 회피할 가능성이 있는 분위기 또는 표현",
    },
    "A-09": {
        "label_id": "COMMUNITY_EMOTIONAL_ESCALATION",
        "name":     "감정 선동 리스크",
        "desc":     "과도한 감정 자극 또는 분노 유도로 과열 반응이 발생할 가능성",
    },
    "A-10": {
        "label_id": "COMMUNITY_PAST_CONTROVERSY",
        "name":     "기존 논란 결합 리스크",
        "desc":     "과거 논란·이슈와 결합되어 추가 확산이 발생할 가능성",
    },
}

# B축 — 플랫폼 리스크 정의
NATAM_B_AXES = {
    "B-01": {
        "label_id": "PLATFORM_HARASSMENT",
        "name":     "괴롭힘·모욕 표현 리스크",
        "desc":     "인신공격·조롱·모욕 표현·지속적 비하 등 플랫폼 정책 위반 가능성",
    },
    "B-02": {
        "label_id": "PLATFORM_VIOLENCE_ILLEGALITY",
        "name":     "폭력·위협·불법행위 리스크",
        "desc":     "폭력 표현·위협 발언·범죄 묘사·위험행동 조장 등",
    },
    "B-03": {
        "label_id": "PLATFORM_HATE_DISCRIMINATION",
        "name":     "혐오·차별 표현 리스크",
        "desc":     "성별·인종·지역 일반화·차별 표현·혐오 밈·사회적 약자 조롱",
    },
    "B-04": {
        "label_id": "PLATFORM_SEXUAL_CONTENT",
        "name":     "성적 표현·대상화 리스크",
        "desc":     "성적 암시·신체 대상화·선정성·성희롱 표현",
    },
    "B-05": {
        "label_id": "PLATFORM_AD_FRIENDLY",
        "name":     "광고친화성·상업 신뢰 리스크",
        "desc":     "과도한 욕설·충격형 썸네일·광고 제한 가능 요소·브랜드 세이프티 충돌",
    },
}

# 단계별 이모지
NATAM_LEVEL_EMOJI = {
    "SAFE":     "🟢",
    "CARE":     "🔵",
    "ALERT":    "🟡",
    "DANGER":   "🟠",
    "CRITICAL": "🔴",
}


def assess_natam_risk(
    client,
    system_instruction: str,
    transcript_text:    str,
    summary:            str,
    gen_model:          str = GEN_MODEL,
) -> dict:
    """
    NATAM Risk Framework v2.0 기반 A축·B축 리스크 평가.

    Gemini에게 콘텐츠 텍스트(자막 요약 + 사건 개요)를 제공하고
    각 축 항목에 대해 SAFE / CARE / ALERT / DANGER / CRITICAL 단계를 반환받음.

    반환 형태:
    {
        "A": {
            "A-01": {"level": "ALERT", "reason": "..."},
            ...
        },
        "B": {
            "B-01": {"level": "SAFE",  "reason": "..."},
            ...
        },
        "overall_a": "ALERT",
        "overall_b": "SAFE",
    }
    """
    a_items_desc = "\n".join(
        f'  "{k}": {v["name"]} — {v["desc"]}'
        for k, v in NATAM_A_AXES.items()
    )
    b_items_desc = "\n".join(
        f'  "{k}": {v["name"]} — {v["desc"]}'
        for k, v in NATAM_B_AXES.items()
    )

    # 분석 텍스트가 너무 길면 앞부분만 사용
    input_text = f"[사건 개요]\n{summary}\n\n[자막 내용 요약]\n{transcript_text[:2000]}"

    prompt = f"""당신은 NATAM Risk Framework v2.0을 적용하는 리스크 분석 전문가입니다.

[분석 원칙]
- 법률 판단, 사실 여부 확정, 도덕적 심판을 하지 않습니다.
- 오직 위험 신호 탐지, 확산 가능성 예측, 플랫폼 충돌 가능성만 분석합니다.
- 단계: SAFE(위험 없음) / CARE(주의) / ALERT(경고) / DANGER(위험) / CRITICAL(심각)

[A축 — 커뮤니티 리스크 항목]
{a_items_desc}

[B축 — 플랫폼 리스크 항목]
{b_items_desc}

[분석 대상 콘텐츠]
{input_text}

위 콘텐츠를 읽고 A축 10개 + B축 5개 항목 각각에 대해 단계와 한 줄 근거를 JSON으로만 반환하세요.

[출력 형식 — JSON만, 설명 없이]
{{
  "A": {{
    "A-01": {{"level": "SAFE", "reason": "정치적 요소 감지되지 않음"}},
    "A-02": {{"level": "CARE", "reason": "특정 유튜버 언급이 반복됨"}},
    ...
  }},
  "B": {{
    "B-01": {{"level": "SAFE", "reason": "모욕 표현 없음"}},
    ...
  }}
}}"""

    # ── NATAM은 리포트에 '항상' 나와야 하므로 평가 실패를 만들지 않는다 ──
    #    ① 정밀 평가(재시도) → ② 간이 종합 평가 폴백 → ③ 보수적 CARE 최후 폴백
    def _call(p: str, max_tokens: int):
        """503/429 일시오류·JSON 파싱 실패 시 재시도. 성공 dict / 실패 None."""
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=gen_model, contents=p,
                    config={
                        "system_instruction": system_instruction,
                        "response_mime_type": "application/json",
                        "temperature":        0.1,
                        "max_output_tokens":  max_tokens,        # 15개 항목 잘림 방지
                        "thinking_config":    {"thinking_budget": 0},  # 추론에 출력예산 낭비 방지
                    },
                )
                parsed = _safe_json_parse(resp.text, "NATAM 리스크 평가")
                if parsed is not None:
                    return parsed
                raise ValueError("JSON 파싱 실패")
            except Exception as e:
                if attempt < 2:
                    wait = 3 * (attempt + 1)   # 3s, 6s (배경 수집 타임아웃 내)
                    print(f"   ⚠️ NATAM 재시도 ({attempt+1}/2): {e} — {wait}초 후")
                    time.sleep(wait)
                    continue
                print(f"   ⚠️ NATAM 호출 실패: {e}")
                return None
        return None

    def _highest(items):
        levels = [v.get("level", "SAFE") for v in items.values()]
        return max(levels, key=lambda l: NATAM_LEVELS.index(l) if l in NATAM_LEVELS else 0)

    # ① 정밀 평가 (항목별)
    result = _call(prompt, 8192)
    if result is not None:
        a_result = result.get("A", {}) or {}
        b_result = result.get("B", {}) or {}
        for k in NATAM_A_AXES:
            if k not in a_result:
                a_result[k] = {"level": "SAFE", "reason": "해당 신호 미감지"}
        for k in NATAM_B_AXES:
            if k not in b_result:
                b_result[k] = {"level": "SAFE", "reason": "해당 신호 미감지"}
        return {"A": a_result, "B": b_result,
                "overall_a": _highest(a_result), "overall_b": _highest(b_result)}

    # ② 간이 종합 평가 폴백 (출력이 작아 성공 확률이 높음)
    print("   ↩️ NATAM 정밀 평가 실패 → 간이 종합 평가로 폴백")
    simple_prompt = (
        "다음 콘텐츠의 NATAM 리스크를 종합 평가하라. A축(커뮤니티 리스크)과 "
        "B축(플랫폼 리스크) 각각의 종합 단계만 SAFE/CARE/ALERT/DANGER/CRITICAL 중 "
        "하나로, 한 줄 근거와 함께 JSON으로만 반환.\n"
        f"[콘텐츠]\n{input_text}\n"
        '[출력] {"overall_a":"CARE","overall_b":"SAFE","reason_a":"...","reason_b":"..."}'
    )
    simple = _call(simple_prompt, 512)
    if simple is not None:
        oa = simple.get("overall_a") if simple.get("overall_a") in NATAM_LEVELS else "CARE"
        ob = simple.get("overall_b") if simple.get("overall_b") in NATAM_LEVELS else "SAFE"
        ra = simple.get("reason_a", "간이 종합 평가")
        rb = simple.get("reason_b", "간이 종합 평가")
        a = {k: {"level": oa, "reason": f"종합 평가 적용 — {ra}"} for k in NATAM_A_AXES}
        b = {k: {"level": ob, "reason": f"종합 평가 적용 — {rb}"} for k in NATAM_B_AXES}
        return {"A": a, "B": b, "overall_a": oa, "overall_b": ob}

    # ③ 최후 폴백 — '평가 실패' 대신 보수적 CARE로 채워 리포트엔 항상 내용이 남게 함
    print("   ⚠️ NATAM 간이 평가도 실패 → 보수적 CARE 적용")
    reason = "자동 평가 일시 지연 — 보수적 주의 단계 적용(수동 확인 권장)"
    a = {k: {"level": "CARE", "reason": reason} for k in NATAM_A_AXES}
    b = {k: {"level": "CARE", "reason": reason} for k in NATAM_B_AXES}
    return {"A": a, "B": b, "overall_a": "CARE", "overall_b": "CARE"}


def _build_natam_placeholders(natam: dict) -> dict:
    """NATAM 평가 결과를 보고서 플레이스홀더 dict로 변환"""
    out = {}

    # A축 개별 항목
    for key, meta in NATAM_A_AXES.items():
        item   = natam.get("A", {}).get(key, {"level": "SAFE", "reason": "—"})
        level  = item.get("level", "SAFE")
        emoji  = NATAM_LEVEL_EMOJI.get(level, "⚪")
        ph_key = key.replace("-", "_")  # A-01 → A_01
        out[f"NATAM_{ph_key}_LEVEL"]  = f"{emoji} {level}"
        out[f"NATAM_{ph_key}_NAME"]   = meta["name"]
        out[f"NATAM_{ph_key}_REASON"] = item.get("reason", "—")

    # B축 개별 항목
    for key, meta in NATAM_B_AXES.items():
        item   = natam.get("B", {}).get(key, {"level": "SAFE", "reason": "—"})
        level  = item.get("level", "SAFE")
        emoji  = NATAM_LEVEL_EMOJI.get(level, "⚪")
        ph_key = key.replace("-", "_")  # B-01 → B_01
        out[f"NATAM_{ph_key}_LEVEL"]  = f"{emoji} {level}"
        out[f"NATAM_{ph_key}_NAME"]   = meta["name"]
        out[f"NATAM_{ph_key}_REASON"] = item.get("reason", "—")

    # 축별 종합
    oa = natam.get("overall_a", "SAFE")
    ob = natam.get("overall_b", "SAFE")
    out["NATAM_OVERALL_A"] = f"{NATAM_LEVEL_EMOJI.get(oa,'⚪')} {oa}"
    out["NATAM_OVERALL_B"] = f"{NATAM_LEVEL_EMOJI.get(ob,'⚪')} {ob}"

    return out


# ════════════════════════════════════════════════════════
# 📥 유튜브 다운로더 & 메타데이터 수집 (yt-dlp)
# ════════════════════════════════════════════════════════

def is_youtube_url(text: str) -> bool:
    return bool(YOUTUBE_URL_PATTERNS.search(text.strip()))


def is_google_drive_url(text: str) -> bool:
    return bool(GOOGLE_DRIVE_URL_PATTERN.search(text.strip()))


def _extract_gdrive_file_id(url: str) -> str | None:
    m = GOOGLE_DRIVE_URL_PATTERN.search(url)
    if m:
        return m.group(1)
    m2 = re.search(r'/d/([\w\-]+)', url)
    return m2.group(1) if m2 else None


def download_google_drive_video(url: str, output_dir: str = "downloads") -> dict:
    """
    Google Drive 공유 링크에서 영상을 다운로드합니다. gdown 라이브러리 사용.
    반환 dict: youtube_meta 와 동일한 키 구조 (확산 지표 제외)
    """
    try:
        import gdown
    except ImportError:
        print("⚠️  gdown 설치 중...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        import gdown

    os.makedirs(output_dir, exist_ok=True)
    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        raise ValueError(f"Google Drive 파일 ID 추출 실패: {url}")

    print(f"📥 Google Drive 다운로드 중 (ID: {file_id}) ...")
    output_path = os.path.join(output_dir, f"gdrive_{file_id}.mp4")

    try:
        gdown.download(id=file_id, output=output_path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"   ⚠️  gdown 기본 시도 실패: {e}  → direct URL 재시도")
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url=direct_url, output=output_path, quiet=False)

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Google Drive 다운로드 실패: {output_path}")

    print(f"✅ Google Drive 다운로드 완료 → {output_path}")
    return {
        "video_path":        output_path,
        "title":             f"Google Drive 영상 ({file_id})",
        "channel":           "Google Drive",
        "view_count":        0,
        "like_count":        0,
        "comment_count":     0,
        "subscriber_count":  0,
        "upload_date":       datetime.now().strftime("%Y%m%d"),
        "duration":          0,
        "url":               url,
        "video_id":          file_id,
        "description":       "",
        "tags":              [],
    }


def download_youtube_video(url: str, output_dir: str = "downloads") -> dict:
    """
    yt-dlp 로 유튜브 영상을 다운로드하고 메타데이터를 반환합니다.

    반환 dict:
      video_path   : 다운로드된 mp4 파일 경로
      title        : 영상 제목
      channel      : 채널명
      view_count   : 조회수 (int)
      like_count   : 좋아요 수 (int)
      comment_count: 댓글 수 (int)
      subscriber_count: 채널 구독자 수 (int)
      upload_date  : 업로드 날짜 (YYYYMMDD)
      duration     : 영상 길이 (초)
      url          : 원본 URL
      video_id     : 유튜브 영상 ID
    """
    try:
        import yt_dlp
    except ImportError:
        print("⚠️  yt-dlp 가 설치되어 있지 않습니다. 설치 중...")
        os.system(f"{sys.executable} -m pip install yt-dlp -q")
        import yt_dlp

    os.makedirs(output_dir, exist_ok=True)

    # ── 1단계: 메타데이터만 먼저 수집 ────────────────────
    print(f"📊 유튜브 메타데이터 수집 중: {url}")
    _cookies_path = '/home/ubuntu/risk-radar/cookies.txt'
    _ydl_base = {
        'quiet': True,
        'no_warnings': True,
        # ── 봇 감지 우회: Android 클라이언트 + 브라우저 User-Agent ──────
        'extractor_args': {
            'youtube': {
                'player_client': ['android_embedded', 'android', 'web'],
            }
        },
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Linux; Android 12; SM-G998B) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/112.0.5615.49 Mobile Safari/537.36'
            ),
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8',
        },
        'sleep_interval':     1,
        'max_sleep_interval': 3,
        # 쿠키 파일이 존재할 때만 사용 (없으면 무시)
        **({"cookiefile": _cookies_path} if os.path.exists(_cookies_path) else {}),
    }
    ydl_opts_meta = {
        **_ydl_base,
        'skip_download': True,
        'extract_flat': False,
    }

    meta = {}
    with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
        info = ydl.extract_info(url, download=False)
        meta = {
            'title':             info.get('title', '제목 없음'),
            'channel':           info.get('channel') or info.get('uploader', '채널 없음'),
            'view_count':        int(info.get('view_count') or 0),
            'like_count':        int(info.get('like_count') or 0),
            'comment_count':     int(info.get('comment_count') or 0),
            'subscriber_count':  int(info.get('channel_follower_count') or 0),
            'upload_date':       info.get('upload_date', 'N/A'),
            'duration':          int(info.get('duration') or 0),
            'url':               url,
            'video_id':          info.get('id', ''),
            'description':       (info.get('description') or '')[:500],
            'tags':              info.get('tags', [])[:10],
        }
        print(f"✅ 메타데이터 수집 완료")
        print(f"   제목: {meta['title']}")
        print(f"   채널: {meta['channel']} (구독자: {meta['subscriber_count']:,}명)")
        print(f"   조회수: {meta['view_count']:,} | 좋아요: {meta['like_count']:,} | 댓글: {meta['comment_count']:,}")
        print(f"   업로드: {meta['upload_date']} | 길이: {meta['duration']}초")

    # ── 2단계: 영상 다운로드 ─────────────────────────────
    safe_title = re.sub(r'[\\/*?:"<>|]', '_', meta['title'])[:60]
    out_template = os.path.join(output_dir, f"{safe_title}_%(id)s.%(ext)s")

    print(f"\n📥 영상 다운로드 중...")
    ydl_opts_dl = {
        **_ydl_base,
        'quiet': False,
        'outtmpl': out_template,
        'format': '18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
        ydl.download([url])

    # 다운로드된 파일 찾기
    video_path = None
    for f in os.listdir(output_dir):
        if meta['video_id'] in f and f.endswith('.mp4'):
            video_path = os.path.join(output_dir, f)
            break

    # video_id 없이 찾는 폴백
    if not video_path:
        for f in sorted(os.listdir(output_dir), key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True):
            if f.endswith('.mp4'):
                video_path = os.path.join(output_dir, f)
                break

    if not video_path:
        raise FileNotFoundError(f"다운로드된 mp4 파일을 찾을 수 없습니다: {output_dir}")

    print(f"✅ 다운로드 완료 → {video_path}")
    meta['video_path'] = video_path
    return meta


# ════════════════════════════════════════════════════════
# 📊 유튜브 메타데이터 기반 확산 단계 판정
# ════════════════════════════════════════════════════════

def analyze_spread_from_youtube_meta(meta: dict) -> dict:
    """
    단일 시점 유튜브 메타데이터로 확산 단계를 판정합니다.
    시계열 데이터가 없으므로 절대값 지표 기반으로 추정합니다.

    판정 기준:
      - CVR (댓글/조회수 비율): 높을수록 논란 가능성↑
      - 좋아요/조회수 비율: 낮을수록 반응이 부정적일 수 있음
      - 조회수 절대값: 채널 규모 대비 과도한 조회수는 외부 유입 가능성
      - 구독자 대비 조회수 비율: 구독자 범위를 벗어나면 외부 확산 신호
    """
    view_count       = meta.get('view_count', 0)
    comment_count    = meta.get('comment_count', 0)
    like_count       = meta.get('like_count', 0)
    subscriber_count = meta.get('subscriber_count', 0)

    reasons  = []
    metrics  = {}

    # ── CVR 계산 ─────────────────────────────────────────
    cvr = (comment_count / view_count * 100) if view_count > 0 else 0
    metrics['cvr'] = round(cvr, 2)

    # ── 좋아요 비율 ───────────────────────────────────────
    like_ratio = (like_count / view_count * 100) if view_count > 0 else 0
    metrics['like_ratio'] = round(like_ratio, 2)

    # ── 구독자 대비 조회수 비율 ───────────────────────────
    sub_view_ratio = (view_count / subscriber_count * 100) if subscriber_count > 0 else 0
    metrics['sub_view_ratio'] = round(sub_view_ratio, 1)

    # ── 실제 수치도 기록 ─────────────────────────────────
    metrics['view_count']       = view_count
    metrics['like_count']       = like_count
    metrics['comment_count']    = comment_count
    metrics['subscriber_count'] = subscriber_count
    metrics['upload_date']      = meta.get('upload_date', 'N/A')

    # ── 단계 판정 로직 ────────────────────────────────────
    stage = "Early"  # 기본값

    # Mid 판정 조건들
    if cvr >= 5.0:
        stage = "Mid"
        reasons.append(f"[유튜브 지표] CVR {cvr:.1f}% ≥ 5% → 댓글 집중 포화 감지")

    if sub_view_ratio >= 200 and subscriber_count > 0:
        stage = "Mid"
        reasons.append(f"[유튜브 지표] 구독자 대비 조회수 {sub_view_ratio:.0f}% → 외부 유입 가능성")

    # Late 판정 조건: 좋아요 비율 매우 낮고 댓글 많음 (논란 후 정착기)
    if like_ratio < 1.0 and cvr >= 3.0 and view_count >= 100000:
        stage = "Late"
        reasons.append(f"[유튜브 지표] 좋아요 비율 {like_ratio:.1f}% 낮고 댓글 집중 → 논란 후기 정착 단계")

    # Early 기본 메시지
    if stage == "Early":
        if view_count < 10000:
            reasons.append(f"[유튜브 지표] 조회수 {view_count:,}회 → 초기 노출 단계")
        else:
            reasons.append(f"[유튜브 지표] 조회수 {view_count:,}회, CVR {cvr:.1f}% → 일반적 확산 범위")

    # 부가 정보 추가
    reasons.append(f"[참고] 좋아요 {like_count:,} / 댓글 {comment_count:,} / 구독자 {subscriber_count:,}명")

    return {
        "stage":   stage,
        "reasons": reasons,
        "metrics": metrics,
        "source":  "youtube_meta",  # 출처 표시 (증가율 계산 불가 구분용)
    }


# ════════════════════════════════════════════════════════
# 📄 CrisisReportEngine (v1.5.1 기반 + youtube_meta 지원)
# ════════════════════════════════════════════════════════

_SPREAD_STAGE_DESC = {
    "Early":   "소수 시청자가 인지한 초기 단계. 빠른 대응으로 확산을 차단할 수 있는 골든타임.",
    "Mid":     "커뮤니티·SNS로 확산이 본격화된 단계. 여론이 형성되고 미디어 관심이 시작됨.",
    "Late":    "대중이 인지한 후기 단계. 추가 확산은 정체되나 이미지 손상이 고착화되는 시점.",
    "Unknown": "확산 지표 데이터 부족으로 단계를 특정할 수 없음.",
}
_LABEL_DESCS = {
    "L01": "명시적 비속어·욕설 포함 발언",
    "L02": "특정인·집단을 대상으로 한 비하 표현",
    "L03": "성별·인종·종교 등 기반 혐오 발언",
    "L04": "사실과 다른 정보의 의도적·비의도적 유포",
    # L05(기만)·L10(저작권)은 별도 Copyright Detector 담당 → 문장 분류 제외
    "L06": "신체적 위험을 초래하거나 조장하는 콘텐츠",
    "L07": "특정 정치 성향의 일방적 주장 또는 선동",
    "L08": "동의 없는 개인정보·사생활 노출",
    "L09": "성적 발언·표현으로 인한 불쾌감 유발",
    "L11": "사건·사고 피해자를 대상으로 한 조롱",
    "L12": "해당 없음",
}
_WORST_FX = {
    "시청자 가스라이팅":             "피해자·비판자의 분노를 증폭시켜 2차 논란 유발",
    "실시간 스트리밍으로 억울함 호소": "즉흥 발언 리스크 및 편집 불가로 추가 실수 가능성 높음",
    "농담이었다 해명":               "진정성 결여로 인식되어 여론 악화 가속",
    "법적 대응 예고":                "외부 기관 개입을 촉발해 사법적 문제로 비화될 위험",
    "채널 비공개":                   "도주 인상을 줘 비판 여론이 오프라인까지 확산될 수 있음",
    "비판 심화 후 채널 비공개":       "이미 악화된 여론에 대한 책임 회피로 인식됨",
    "솔직한 후기였다 강변":           "실제 피해자 증언으로 반박당할 경우 신뢰도 완전 붕괴",
    "편집 오류 핑계":                "의도성 논란으로 번져 더 큰 불신 초래",
}
_DEFAULT_ACTIONS = {
    "Early": {
        "immediate": ["문제 구간 내부 검토 및 증거 보전", "커뮤니티·SNS 언급량 모니터링 시작"],
        "short":     ["법률·PR 전문가와 대응 방향 사전 협의", "공식 입장문 초안 작성 (공개 여부는 상황 판단)"],
        "mid":       ["구독자 대상 현황 안내 여부 검토", "재발 방지 내부 프로세스 점검"],
    },
    "Mid": {
        "immediate": ["문제 콘텐츠 접근 제한 또는 수정본 업로드 검토", "공식 입장문 즉시 작성 및 채널 공지"],
        "short":     ["커뮤니티 댓글·DM 모니터링 및 악성 댓글 대응 방침 수립", "미디어 문의 대응 창구 일원화"],
        "mid":       ["구독자 신뢰 회복을 위한 콘텐츠 기획", "유사 상황 재발 방지 프로세스 문서화"],
    },
    "Late": {
        "immediate": ["추가 악화 요인 차단 (불필요한 발언·라이브 자제)", "기존 공개 입장문 일관성 유지"],
        "short":     ["장기적 이미지 회복 전략 수립", "법적 리스크 최종 점검"],
        "mid":       ["채널 방향성 재정립 및 콘텐츠 로드맵 수정", "구독자 이탈 분석 및 피드백 수렴"],
    },
    "Unknown": {
        "immediate": ["확산 지표 데이터 수집 우선 진행", "내부 상황 파악 및 관계자 보고"],
        "short":     ["전문가 자문 의뢰", "모니터링 체계 구축"],
        "mid":       ["데이터 확보 후 단계별 대응 전환", "재발 방지 검토"],
    },
}


def _rp_safe(val, default="N/A"):
    return str(val).strip() if val not in (None, "", [], {}) else default

def _build_spread_arrow_line(stage: str) -> str:
    """확산 단계에 따라 ▲ 위치를 이동시킨 박스 한 줄을 반환 (너비 56자 고정)"""
    # 박스 내부 콘텐츠 54자, ▲의 0-based 위치 (Early=5, Mid=20, Late=35, Unknown=27)
    positions = {"Early": 5, "Mid": 20, "Late": 35, "Unknown": 27}
    pos = positions.get(stage, positions["Unknown"])
    inner = " " * pos + "▲" + " " * (54 - pos - 1)
    return f"║{inner}║"

def _fmt_ts(sec) -> str:
    """초 → mm:ss (없으면 '—')"""
    try:
        s = int(float(sec))
    except (TypeError, ValueError):
        return "—"
    if s < 0:
        return "—"
    return f"{s // 60:02d}:{s % 60:02d}"


def _seg_timestamp(seg: dict) -> str:
    """세그먼트의 start~end를 'mm:ss–mm:ss'로. start 없으면 '—'"""
    if seg.get("start") is None:
        return "—"
    return f"{_fmt_ts(seg.get('start'))}–{_fmt_ts(seg.get('end'))}"


def _build_transcript_rows(ta: list) -> str:
    """주요 발언 테이블 행 반환(타임스탬프 포함).
    L12(해당 없음)와 L04(허위 정보)는 제외 — L04는 별도 '검증 필요 주장(Stage1)' 섹션이 담당."""
    rows = [
        s for s in ta
        if not s.get("label", "").startswith(("L12", "L04"))
    ]
    if not rows:
        return "| — | — | (감지된 주요 발언 없음) | — | — |"
    lines = []
    for i, s in enumerate(rows, 1):
        ts     = _rp_safe(s.get("timestamp"), "—")
        text   = _rp_safe(s.get("corrected_text") or s.get("text"), "—")
        label  = _rp_safe(s.get("label"), "—")
        reason = _rp_safe(s.get("reason"), "—")
        # 파이프 문자가 셀을 깨지 않도록 이스케이프
        text   = text.replace("|", "\\|")
        reason = reason.replace("|", "\\|")
        lines.append(f"| {i} | {ts} | {text} | {label} | {reason} |")
    return "\n".join(lines)


def _build_claim_check_rows(claims: list) -> str:
    """[L04 Stage1] '검증 필요 주장'을 마크다운 테이블 행으로 (진위 판정은 포함하지 않음)."""
    if not claims:
        return "| — | — | (검증이 필요한 사실 주장 없음) | — | — |"
    _cw = {"HIGH": "🔴 높음", "MEDIUM": "🟡 중간", "LOW": "🟢 낮음"}
    lines = []
    for i, c in enumerate(claims, 1):
        ts     = _rp_safe(c.get("timestamp"), "—")
        claim  = _rp_safe(c.get("claim"), "—").replace("|", "\\|")
        domain = _rp_safe(c.get("domain"), "—")
        cw     = _cw.get(c.get("checkworthiness"), c.get("checkworthiness") or "—")
        reason = _rp_safe(c.get("reason"), "—").replace("|", "\\|")
        lines.append(f"| {i} | {ts} | {claim} | {domain} / {cw} | {reason} |")
    return "\n".join(lines)


def _safe_json_parse(text: str, context: str = "") -> object:
    """
    JSON 파싱 유틸.
    실패 시 ①마크다운 블록 제거 → ②배열/오브젝트 추출 → ③불완전 배열 절단 복구 순서로 재시도.
    Gemini가 긴 응답을 중간에 잘라버릴 때(Unterminated string 에러) 복구에 사용.
    """
    if not text:
        return None

    # 1) 직접 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) 마크다운 코드블록 제거 후 재시도
    stripped = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3) 배열/오브젝트 패턴 직접 추출
    for pat in [r'(\[[\s\S]+\])', r'(\{[\s\S]+\})']:
        m = re.search(pat, stripped)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

    # 4) 불완전 배열 복구: 마지막으로 완성된 객체( }, ) 까지만 잘라내기
    if stripped.lstrip().startswith('['):
        last = stripped.rfind('},')
        if last > 0:
            try:
                recovered = json.loads(stripped[:last + 1] + ']')
                tag = f" ({context})" if context else ""
                print(f"  ⚠️  JSON 절단 감지{tag} → {len(recovered)}개 객체로 복구 (원본 손실 없음)")
                return recovered
            except Exception:
                pass

    if context:
        print(f"  ❌ JSON 파싱 최종 실패 ({context}): {text[:120]}...")
    return None


def _salvage_segments(text: str) -> list:
    """
    전사 응답({"segments":[...]})이 출력 토큰 한도로 잘렸을 때,
    완성된 세그먼트 객체만 정규식으로 추출해 복구한다(마지막 미완성 객체는 버림).
    """
    seg_re = re.compile(
        r'\{\s*"start"\s*:\s*([0-9.]+)\s*,\s*"end"\s*:\s*([0-9.]+)\s*,'
        r'\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    )
    out = []
    for m in seg_re.finditer(text):
        try:
            txt = json.loads('"' + m.group(3) + '"')   # 이스케이프 해제
        except Exception:
            txt = m.group(3)
        out.append({"start": float(m.group(1)), "end": float(m.group(2)), "text": txt})
    return out


def _rp_eval(value, threshold, over, under):
    try:    return over if float(value) >= threshold else under
    except: return "데이터 없음"

def _rp_label_desc(label_str):
    return _LABEL_DESCS.get(label_str.strip()[:3].upper(), "정의 없음")

def _rp_worst_fx(action):
    for key, fx in _WORST_FX.items():
        if key in action: return fx
    return "여론 악화 가능성 있음"

def _rp_parse_pattern(text):
    parts = re.split(r'\n\s*(?:\d+\.|#+|\*\*)\s*', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    while len(parts) < 3:
        parts.append("사례 데이터 부족으로 분석 불가")
    return parts[0], parts[1], parts[2]

def _save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class CrisisReportEngine:
    """report template.md 를 채워 FINAL_REPORT_{ID}_{timestamp}.md 생성"""

    def __init__(self, template_path="report template.md", output_dir="reports"):
        self.template_path = template_path
        self.output_dir    = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def create_report(self, report: dict) -> str | None:
        try:
            if not os.path.exists(self.template_path):
                print(f"❌ 템플릿 파일이 없습니다: {self.template_path}")
                return None
            with open(self.template_path, "r", encoding="utf-8") as f:
                content = f.read()

            data_map = self._build_data_map(report)
            for key, value in data_map.items():
                content = content.replace("{" + key + "}", str(value))

            remaining = re.findall(r'\{[A-Z_0-9]+\}', content)
            if remaining:
                print(f"⚠️  미치환 플레이스홀더 {len(remaining)}개: {remaining[:5]}")

            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"FINAL_REPORT_{data_map['INCIDENT_ID']}_{ts}.md"
            path     = os.path.join(self.output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ MD 리포트 생성 완료: {path}")
            return path
        except Exception as e:
            print(f"❌ MD 리포트 생성 실패: {e}")
            return None

    def _build_data_map(self, report: dict) -> dict:
        meta   = report.get("meta", {})
        spread = report.get("spread_stage", {})
        cls    = report.get("classification", {})
        rule   = report.get("rule_scan", {})
        cases  = report.get("similar_cases", [])
        ta     = report.get("transcript_analysis", [])
        claims = (report.get("claim_check", {}) or {}).get("claims", [])  # [L04 Stage1]
        kf     = report.get("keyframe_analysis", [])
        wa     = report.get("worst_actions", [])
        tf     = report.get("transcript_files", {})
        pat    = report.get("pattern_summary", "")
        yt     = report.get("youtube_meta", {})  # 유튜브 메타데이터
        natam  = report.get("natam_risk", {})    # NATAM v2.0 리스크

        incident_id = (
            meta.get("video_filename","").replace(".","_").replace(" ","_")
            or datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        stage    = spread.get("stage", "Unknown")
        metrics  = spread.get("metrics") or {}
        labels   = cls.get("labels", [])
        rule_hit = rule.get("hit", False)
        wa_padded = (wa + ["—","—","—","—"])[:4]
        p_s, p_r, p_t = _rp_parse_pattern(pat)
        actions  = _DEFAULT_ACTIONS.get(stage, _DEFAULT_ACTIONS["Unknown"])
        label_1  = labels[0] if labels else "—"
        label_2  = labels[1] if len(labels) > 1 else "—"

        # ── 3-2 판정 지표: 유튜브 URL 입력인지 로컬 파일인지 구분 ──
        is_yt_meta = spread.get("source") == "youtube_meta"

        if is_yt_meta:
            # 유튜브 실측 데이터
            view_count     = metrics.get('view_count', 0)
            comment_count  = metrics.get('comment_count', 0)
            like_count     = metrics.get('like_count', 0)
            sub_count      = metrics.get('subscriber_count', 0)
            cvr            = metrics.get('cvr', 0.0)
            like_ratio     = metrics.get('like_ratio', 0.0)
            sub_view_ratio = metrics.get('sub_view_ratio', 0.0)

            spread_table_rows = (
                f"| 조회수 (실측) | {view_count:,}회 | — | "
                f"좋아요 {like_count:,}회 / 좋아요율 {like_ratio:.1f}% |\n"
                f"| 댓글 수 (실측) | {comment_count:,}건 | — | "
                f"댓글수 {comment_count:,}건 |\n"
                f"| 댓글/조회 비율 (CVR) | {cvr:.2f}% | ≥ 5% → Mid | "
                f"{_rp_eval(cvr, 5.0, '⚠️ 집중 포화 감지', '✅ 일반 범위')} |\n"
                f"| 구독자 대비 조회수 | {sub_view_ratio:.0f}% | ≥ 200% → Mid | "
                f"{'⚠️ 외부 유입 가능성' if sub_view_ratio >= 200 else '✅ 정상'} |"
            )
        else:
            # 로컬 파일 — 유튜브 지표 행 자체를 제거하고 "데이터 없음" 안내만 표시
            spread_table_rows = (
                "| (지표 없음) | 로컬 영상 분석 시 유튜브 실측 지표를 수집할 수 없습니다. | — | — |"
            )

        has_dr = any(s.get("label","").startswith(("L01","L02","L03")) for s in ta)
        has_vr = any(
            any(kw in _rp_safe(kf[i].get("tag","")) for kw in ["욕설","폭력","노출","위험"])
            for i in range(len(kf))
        )

        sr = "\n".join(
            f"- {r}" for r in spread.get("reasons", ["지표 데이터 없음 → 기본값 적용"])
        )

        # 유튜브 채널 정보 블록 (URL 입력 시에만)
        yt_info_block = ""
        if yt:
            yt_info_block = (
                f"\n\n**[유튜브 원본 정보]**\n"
                f"- 채널: {yt.get('channel', 'N/A')} (구독자 {yt.get('subscriber_count', 0):,}명)\n"
                f"- 업로드: {yt.get('upload_date', 'N/A')}\n"
                f"- 원본 URL: {yt.get('url', 'N/A')}"
            )

        def sv(i, k, d="—"): return _rp_safe(ta[i].get(k,d),d) if i < len(ta) else d
        def kv(i, k, d="—"): return _rp_safe(kf[i].get(k,d),d) if i < len(kf) else d
        def cv(i, k, d="—"): return _rp_safe(cases[i].get(k,d),d) if i < len(cases) else d
        def cr(i):
            return ", ".join(cases[i].get("response_pattern",[])) if i < len(cases) else "—"

        base_map = {
            "INCIDENT_ID": incident_id,
            "ANALYZED_AT": _rp_safe(meta.get("analyzed_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
            "VIDEO_FILENAME": _rp_safe(meta.get("video_filename", meta.get("input_query","텍스트 입력"))),
            "INCIDENT_TITLE": _rp_safe(
                meta.get("incident_title") or meta.get("video_filename") or meta.get("input_query","—")
            ),
            "INPUT_QUERY": _rp_safe(
                (
                    f"[{meta['incident_title']}] {meta['incident_summary']}"
                    if meta.get("incident_title") and meta.get("incident_summary")
                    else meta.get("input_query")
                ) or meta.get("video_filename", "—")
            ),
            "DURATION_SEC":    _rp_safe(meta.get("duration_sec","—")),
            "ELAPSED_SEC":     _rp_safe(meta.get("elapsed_sec","—")),
            "TOTAL_SEGMENTS":  _rp_safe(meta.get("total_segments","—")),
            "TOTAL_CORRECTED": _rp_safe(meta.get("total_corrected","—")),
            "PRIMARY_LABEL":   _rp_safe(cls.get("primary","")),
            "SPREAD_STAGE":      stage,
            "SPREAD_ARROW_LINE": _build_spread_arrow_line(stage),
            "INCIDENT_SUMMARY_TEXT": _rp_safe(
                (
                    f"[{meta['incident_title']}] {meta['incident_summary']}"
                    if meta.get("incident_title") and meta.get("incident_summary")
                    else meta.get("incident_summary")
                       or meta.get("input_query")
                       or "자막 전체 텍스트 기반 자동 분석 결과입니다."
                )
            ) + yt_info_block,
            "TRANSCRIPT_ROWS": _build_transcript_rows(ta),
            "CLAIM_CHECK_ROWS":  _build_claim_check_rows(claims),
            "CLAIM_CHECK_COUNT": str(len(claims)),
            "LABEL_1": label_1, "LABEL_1_DESC": _rp_label_desc(label_1),
            "LABEL_2": label_2, "LABEL_2_DESC": _rp_label_desc(label_2),
            "CLASSIFICATION_REASON": _rp_safe(cls.get("reason","—")),
            "RULE_HIT_STATUS":   "🚨 키워드 적발" if rule_hit else "✅ 즉각 위험 없음",
            "RULE_POLICY":       _rp_safe(rule.get("policy","해당 없음")),
            "RULE_SEVERITY":     _rp_safe(rule.get("severity","—")),
            "RULE_MATCHED_WORD": _rp_safe(rule.get("matched_word","—")),
            "RULE_ACTION":       _rp_safe(rule.get("action","—")),
            "RULE_RISK":         "🚨 적발됨" if rule_hit else "✅ 없음",
            "KEYFRAME_1_TIMESTAMP": kv(0,"time","Point 1"), "KEYFRAME_1_TAG": kv(0,"tag"),
            "KEYFRAME_2_TIMESTAMP": kv(1,"time","Point 2"), "KEYFRAME_2_TAG": kv(1,"tag"),
            "KEYFRAME_3_TIMESTAMP": kv(2,"time","Point 3"), "KEYFRAME_3_TAG": kv(2,"tag"),
            "DIRECT_SPEECH_RISK": "🚨 감지됨" if has_dr else "✅ 없음",
            "VISUAL_RISK":        "🚨 감지됨" if has_vr else "✅ 없음",
            "EXTERNAL_SIGNAL":    "✅ 없음",
            "SPREAD_STAGE_DESCRIPTION": _SPREAD_STAGE_DESC.get(stage,"—"),
            # 3-2 테이블: 입력 방식에 따라 동적 생성된 행
            "SPREAD_TABLE_ROWS": spread_table_rows,
            "SPREAD_REASONS": sr,
            "DETECTED_TYPE": cases[0].get("controversy_type","N/A") if cases else "N/A",
            "WORST_ACTION_1": wa_padded[0], "WORST_ACTION_1_EFFECT": _rp_worst_fx(wa_padded[0]),
            "WORST_ACTION_2": wa_padded[1], "WORST_ACTION_2_EFFECT": _rp_worst_fx(wa_padded[1]),
            "WORST_ACTION_3": wa_padded[2], "WORST_ACTION_3_EFFECT": _rp_worst_fx(wa_padded[2]),
            "WORST_ACTION_4": wa_padded[3], "WORST_ACTION_4_EFFECT": _rp_worst_fx(wa_padded[3]),
            "TRIGGER_SUMMARY": p_t,
            "CASE_1_TITLE": cv(0,"title"), "CASE_1_TYPE": cv(0,"controversy_type"), "CASE_1_DISTANCE": cv(0,"distance"), "CASE_1_RESPONSE": cr(0), "CASE_1_OUTCOME": cv(0,"outcome","데이터 없음"),
            "CASE_2_TITLE": cv(1,"title"), "CASE_2_TYPE": cv(1,"controversy_type"), "CASE_2_DISTANCE": cv(1,"distance"), "CASE_2_RESPONSE": cr(1), "CASE_2_OUTCOME": cv(1,"outcome","데이터 없음"),
            "CASE_3_TITLE": cv(2,"title"), "CASE_3_TYPE": cv(2,"controversy_type"), "CASE_3_DISTANCE": cv(2,"distance"), "CASE_3_RESPONSE": cr(2), "CASE_3_OUTCOME": cv(2,"outcome","데이터 없음"),
            "PATTERN_SPREAD_PATH": p_s, "PATTERN_RESPONSE_REACTION": p_r, "PATTERN_TRIGGER": p_t,
            "ACTION_IMMEDIATE_1": actions["immediate"][0], "ACTION_IMMEDIATE_2": actions["immediate"][1],
            "ACTION_SHORT_1":     actions["short"][0],     "ACTION_SHORT_2":     actions["short"][1],
            "ACTION_MID_1":       actions["mid"][0],       "ACTION_MID_2":       actions["mid"][1],
            "PRIORITY_HIGH_ACTION": actions["immediate"][0], "PRIORITY_HIGH_EFFECT": "빠른 대응으로 확산 차단 가능",     "PRIORITY_HIGH_RISK": "섣부른 공개 대응 시 역풍 가능",
            "PRIORITY_MID_ACTION":  actions["short"][0],     "PRIORITY_MID_EFFECT":  "전문가 협의를 통한 리스크 최소화", "PRIORITY_MID_RISK":  "대응 지연 시 여론 주도권 상실",
            "PRIORITY_LOW_ACTION":  actions["mid"][0],       "PRIORITY_LOW_EFFECT":  "장기적 신뢰 회복 기반 마련",       "PRIORITY_LOW_RISK":  "단기 효과 미미할 수 있음",
            "EFFECTIVE_RESPONSE_PATTERNS": (
                "유사 사례에서 초기 투명한 사실 관계 인정이 여론 악화를 줄이는 패턴이 관찰되었습니다. "
                "반면 축소·부인·법적 대응 예고는 오히려 위기를 심화시키는 경향이 있습니다."
            ),
            "TRANSCRIPT_RAW_PATH":        _rp_safe(tf.get("raw","—")),
            "TRANSCRIPT_NORMALIZED_PATH": _rp_safe(tf.get("normalized","—")),
            "TRANSCRIPT_REFINED_PATH":    _rp_safe(tf.get("refined","—")),
            "REPORT_JSON_PATH": os.path.join(
                "reports",
                f"report_{_rp_safe(meta.get('video_filename','unknown')).replace('.','_')}.json"
            ),
            "WHISPER_MODEL":       WHISPER_MODEL,
            "GEN_MODEL":           GEN_MODEL,
            "GEMINI_REFINE_MODEL": GEMINI_REFINE_MODEL,
            "EMBED_MODEL":         EMBED_MODEL,
            "CASE_DB_SIZE":        "30",
            "GEMINI_BATCH_SIZE":   "20",
        }

        # NATAM v2.0 플레이스홀더 병합
        base_map.update(_build_natam_placeholders(natam))
        return base_map


# ════════════════════════════════════════════════════════
# 🔧 유틸 함수
# ════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'(ㅋ|ㅎ|ㅠ|ㅜ|!|\?|\.){3,}', r'\1\1', text)
    text = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣!?.,]', '', text)
    for short, full in ABBREVIATION_MAP.items():
        text = text.replace(short, full)
    return re.sub(r'\s+', ' ', text).strip()

def load_rules(yaml_path):
    if not os.path.exists(yaml_path): return {}
    try:
        with open(yaml_path,'r',encoding='utf-8') as f: lines = f.readlines()
        fixed = [l.lstrip() if not l.lstrip().startswith('#') else l.lstrip() for l in lines]
        raw = yaml.safe_load(''.join(fixed))
    except: return {}
    if not isinstance(raw, dict): return {}
    policies = raw.get('risk_policies', raw)
    return policies if isinstance(policies, dict) else {}

def rule_engine(text, rules):
    for pname, detail in rules.items():
        if not isinstance(detail, dict): continue
        for word in detail.get('keywords', []):
            if word in text:
                return {"hit":True,"policy":pname,"action":detail.get('action','REVIEW'),
                        "severity":detail.get('severity','UNKNOWN'),"matched_word":word}
    return {"hit": False}

def load_controversy_labels(yaml_path):
    if not os.path.exists(yaml_path): return []
    with open(yaml_path,'r',encoding='utf-8') as f: data = yaml.safe_load(f)
    return data.get('labels',[])

def format_labels_for_prompt(labels):
    return ", ".join([f"{l['id']} {l['name']}" for l in labels if l.get('id') != 'L12'])


class SpreadAnalyzer:
    """기존 시계열 데이터 기반 확산 판정 (로컬 파일 분석용)"""
    def calculate_metrics(self, case_data):
        meta     = case_data.get("incident_metadata",{})
        views    = meta.get("view_counts",[])
        comments = meta.get("comment_count_1h",[])
        if not isinstance(views,list) or len(views) < 2: return None
        vg  = ((views[-1]-views[-2])/views[-2])*100
        cg  = ((comments[-1]-comments[-2])/comments[-2])*100 if len(comments)>=2 else 0
        cvr = (comments[-1]/views[-1])*100 if views[-1] else 0
        return {"v_growth":vg,"c_growth":cg,"cvr":cvr}

    def predict_stage(self, incident_data):
        m = self.calculate_metrics(incident_data)
        if not m: return {"stage":"Unknown","reasons":["데이터 부족"],"metrics":None}
        reasons = []
        if m["v_growth"]<5 and m["c_growth"]<5:
            stage="Late";  reasons.append("조회수·댓글 증가율 5% 미만 → 정체기")
        elif m["v_growth"]>=40 or m["c_growth"]>=50:
            stage="Mid";   reasons.append("증가율 임계치 초과 (V≥40% or C≥50%)")
        else:
            stage="Early"; reasons.append("초기 유입 및 참여 감지")
        if m["cvr"]>=5.0:
            stage="Mid"; reasons.append("[Override] CVR≥5% → 소수 집중 포화 감지")
        if incident_data.get("stats",{}).get("external_links",0)>=10 and stage=="Early":
            stage="Mid"; reasons.append("[Override] 외부 커뮤니티 유입 신호 강함")
        return {"stage":stage,"reasons":reasons,"metrics":m}


class RiskConsultant:
    def __init__(self, yaml_path):
        self.rules = {}
        if os.path.exists(yaml_path):
            with open(yaml_path,'r',encoding='utf-8') as f:
                self.rules = yaml.safe_load(f).get('controversy_rules',{})

    def get_worst_actions(self, controversy_type, spread_stage):
        sk      = spread_stage.lower()
        c_rules = self.rules.get(controversy_type,{})
        actions = c_rules.get(sk,[])
        if not actions and sk != 'early': actions = c_rules.get('early',[])
        return actions

    def polish_tone(self, raw_actions):
        reps = [("하는 것","하는 대응"),("삭제","정리하는 행위"),
                ("조롱","희화화하는 반응"),("거짓말","사실과 다른 주장")]
        polished = []
        for a in raw_actions:
            for old,new in reps: a = a.replace(old,new)
            polished.append(a)
        return polished


# ════════════════════════════════════════════════════════
# 🧠 핵심 시스템 클래스
# ════════════════════════════════════════════════════════

class CrisisConsultantSystem:
    def __init__(
        self,
        db_path                = 'case_db.json',
        rules_yaml             = 'rules.yaml',
        labels_yaml            = 'controversy_labels.yaml',
        worst_actions_yaml     = 'worst_actions_map.yaml',
        template_path          = 'report template.md',
        report_dir             = 'reports',
        controversy_model_path = CONTROVERSY_MODEL_PATH,
    ):
        self.client = genai.Client(api_key=API_KEY)
        self.cost   = CostTracker()   # 분석 1건당 토큰/비용 계측

        if TRANSCRIBE_ENGINE == "whisper":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔄 Whisper 모델 로딩 중... (장치: {self.device.upper()})")
            self.whisper_model = whisper.load_model(WHISPER_MODEL).to(self.device)
        elif TRANSCRIBE_ENGINE == "clova":
            self.whisper_model = None
            if CLOVA_INVOKE_URL and CLOVA_SECRET_KEY:
                print("🤖 전사 엔진: Clova NEST (한국어 CER 우위 검증 모드)")
            else:
                print("⚠️  TRANSCRIBE_ENGINE=clova 이지만 CLOVA_INVOKE_URL/SECRET_KEY 미설정 → Gemini 폴백")
        else:
            self.whisper_model = None
            print("🤖 전사 엔진: Gemini Files API (Whisper 모델 로드 생략)")

        with open(db_path, 'r', encoding='utf-8') as f:
            self.cases = json.load(f)

        self.rules            = load_rules(rules_yaml)
        self.labels           = load_controversy_labels(labels_yaml)
        self.label_prompt_str = format_labels_for_prompt(self.labels)
        self.risk_consultant  = RiskConsultant(worst_actions_yaml)
        self.spread_analyzer  = SpreadAnalyzer()
        self.report_dir       = report_dir
        self.report_engine    = CrisisReportEngine(template_path, report_dir)

        self.system_instruction = (
            "당신은 크리에이터 위기 관리 사례 분석 전문가입니다.\n"
            "1. '~하세요', '해야 합니다' 같은 명령·추천은 절대 하지 않습니다.\n"
            "2. '~하는 경향이 있음', '~한 패턴이 관찰됨'처럼 객관적 사실 위주로 서술합니다.\n"
            "3. 반드시 제공된 사례 데이터에 근거해서만 답변합니다."
        )

        # ── FAISS 벡터 인덱스 빌드 (캐시 우선, 없으면 병렬 임베딩) ──
        embed_cache_path = os.path.splitext(db_path)[0] + "_embeddings.npy"
        print("⏳ 벡터 DB 빌드 중...")
        if os.path.exists(embed_cache_path):
            self.embeddings_np = np.load(embed_cache_path).astype('float32')
            print(f"   📦 임베딩 캐시 로드 → {embed_cache_path}")
        else:
            def _embed_one(case):
                resp = self.client.models.embed_content(
                    model=EMBED_MODEL, contents=case['summary'],
                )
                return resp.embeddings[0].values

            with ThreadPoolExecutor(max_workers=8) as ex:
                embeddings = list(ex.map(_embed_one, self.cases))
            self.embeddings_np = np.array(embeddings).astype('float32')
            np.save(embed_cache_path, self.embeddings_np)
            print(f"   💾 임베딩 캐시 저장 → {embed_cache_path}")

        self.index = faiss.IndexFlatL2(self.embeddings_np.shape[1])
        self.index.add(self.embeddings_np)

        # ── ML 논란 분류 모델 ─────────────────────────────────
        try:
            _bundle       = joblib.load(controversy_model_path)
            self.ml_model = _bundle['model']
            self.ml_mlb   = _bundle['mlb']
            print(f"✅ ML 분류 모델 로드 완료 (클래스 {len(self.ml_mlb.classes_)}개: {list(self.ml_mlb.classes_)})")
        except Exception as _e:
            print(f"⚠️  ML 분류 모델 로드 실패: {_e} → Gemini 분류로 대체")
            self.ml_model = None
            self.ml_mlb   = None

        print(f"✅ 시스템 준비 완료! (사례 DB: {len(self.cases)}건)")

    # ────────────────────────────────────────────────────
    # 🎙️ Whisper 전사
    # ────────────────────────────────────────────────────
    def transcribe(self, video_path: str) -> dict:
        print(f"🎙️  전사 시작: {video_path}")
        result = self.whisper_model.transcribe(
            video_path,
            verbose=False,
            beam_size=2,   # 5→2: 정확도 소폭 감소, 속도 약 2배 향상
            best_of=2,     # 5→2: 마찬가지
            fp16=(self.device == "cuda"),
        )
        return {
            "video_info": {
                "path":     video_path,
                "duration": round(result['segments'][-1]['end'], 2) if result['segments'] else 0,
            },
            "segments": result["segments"],
        }

    # ────────────────────────────────────────────────────
    # 🔁 Gemini 전사 실패 시 로컬 Whisper 폴백
    # ────────────────────────────────────────────────────
    def _ensure_whisper_model(self):
        """폴백용 Whisper 모델을 필요한 시점에 1회만 로드(엔진이 gemini/clova여도 동작)."""
        if getattr(self, "whisper_model", None) is not None:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   🔄 Whisper 폴백 모델 로딩 중... (장치: {self.device.upper()}, 모델: {WHISPER_MODEL})")
        self.whisper_model = whisper.load_model(WHISPER_MODEL).to(self.device)

    def _transcribe_with_whisper(self, video_path: str) -> dict:
        """Whisper 모델 보장 후 로컬 전사 수행(네트워크 의존 없음)."""
        self._ensure_whisper_model()
        return self.transcribe(video_path)

    # ────────────────────────────────────────────────────
    # 🤖 Gemini Files API 전사
    # ────────────────────────────────────────────────────
    def transcribe_with_gemini(self, video_path: str) -> dict:
        """Gemini Files API로 영상을 업로드하고 전사 결과를 Whisper 동일 형식으로 반환"""
        print(f"🤖 Gemini 전사 시작: {video_path}")

        print("   📤 파일 업로드 중...")
        # 파일명에 한글이 있으면 HTTP 헤더 ASCII 인코딩 오류 발생 →
        # 파일을 바이너리 스트림으로 열고 ASCII 안전 display_name 지정
        import mimetypes as _mt
        _mime  = _mt.guess_type(video_path)[0] or "video/mp4"
        _ext   = os.path.splitext(video_path)[1].lower() or ".mp4"
        with open(video_path, "rb") as _fh:
            file_ref = self.client.files.upload(
                file=_fh,
                config={"display_name": f"upload{_ext}", "mime_type": _mime},
            )

        # ACTIVE 상태까지 대기
        while file_ref.state.name == "PROCESSING":
            print("   ⏳ 파일 처리 중...")
            time.sleep(5)
            file_ref = self.client.files.get(name=file_ref.name)

        if file_ref.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini 파일 처리 실패 (state={file_ref.state.name})")

        print("   ✅ 업로드 완료, 전사 요청 중...")

        prompt = (
            "이 영상/오디오를 전사해줘. 아래 JSON 형식으로만 반환해줘 (설명 없이).\n\n"
            '{"segments":[{"start":0.0,"end":5.2,"text":"전사 텍스트"},...], "duration":전체길이초}\n\n'
            "[규칙]\n"
            "- start/end는 초 단위 실수\n"
            "- 발화 단위(문장·호흡)로 세그먼트 분리\n"
            "- 묵음 구간 제외\n"
            "- 한국어·영어 혼용 그대로 전사"
        )

        # 503(과부하)/429 등 일시적 서버 오류 대비 자동 재시도(지수 백오프)
        resp = None
        last_err = None
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=GEN_MODEL,
                    contents=[file_ref, prompt],
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.0,
                        "max_output_tokens": 65536,            # 긴 전사 잘림 방지
                        "thinking_config": {"thinking_budget": 0},  # 추론에 출력예산 낭비 방지
                    },
                )
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                transient = ("503" in msg or "UNAVAILABLE" in msg
                             or "overloaded" in msg.lower() or "high demand" in msg.lower()
                             or "429" in msg or "500" in msg or "deadline" in msg.lower())
                if transient and attempt < max_retries - 1:
                    wait = min(5 * (2 ** attempt), 40)   # 5,10,20,40,40 백오프
                    print(f"   ⚠️ Gemini 일시 오류(과부하/혼잡) — {wait}초 후 재시도 ({attempt+1}/{max_retries-1})")
                    time.sleep(wait)
                    continue
                break   # 비일시 오류이거나 재시도 소진 → 루프 종료 후 폴백 처리

        # 업로드 파일 정리(성공/실패 공통)
        try:
            self.client.files.delete(name=file_ref.name)
        except Exception:
            pass

        if resp is None:
            # Gemini 전사 최종 실패 → 로컬 Whisper로 폴백하여 분석이 중단되지 않게 함
            print(f"   ⚠️ Gemini 전사 실패({last_err}) → 로컬 Whisper 전사로 폴백합니다")
            try:
                return self._transcribe_with_whisper(video_path)
            except Exception as we:
                raise RuntimeError(
                    f"Gemini 전사 실패 후 Whisper 폴백도 실패: {we} (원인: {last_err})"
                ) from we

        self.cost.add(resp, "transcribe")

        result   = _safe_json_parse(resp.text, "Gemini 전사")
        duration = None

        # Gemini가 형식을 흔든다: ① {"segments":[...]} ② [ {..},{..} ] (최상위 배열)
        #                        ③ [ {"segments":[...]} ] (객체를 배열로 한 겹 감쌈)
        if isinstance(result, dict):
            segments = result.get("segments", [])
            if "duration" in result:
                try:    duration = float(result["duration"])
                except (TypeError, ValueError): duration = None
        elif isinstance(result, list):
            segments = result            # 최상위가 세그먼트 배열인 경우
        else:
            # 출력 토큰 한도로 JSON이 잘린 경우 → 완성된 세그먼트만 정규식으로 복구
            segments = _salvage_segments(resp.text or "")
            if segments:
                print(f"   ⚠️ 전사 응답이 잘렸지만 {len(segments)}개 세그먼트 복구 성공 (뒷부분 일부 누락 가능)")
            else:
                raise ValueError("Gemini 전사 JSON 파싱 실패 — 응답: " + (resp.text[:200] if resp.text else "빈 응답"))

        # ③ 배열 안에 {"segments":[...]} 가 한 겹 더 들어온 경우 풀어줌
        if (len(segments) == 1 and isinstance(segments[0], dict)
                and isinstance(segments[0].get("segments"), list)):
            inner    = segments[0]
            segments = inner["segments"]
            if duration is None and "duration" in inner:
                try:    duration = float(inner["duration"])
                except (TypeError, ValueError): duration = None

        # 세그먼트가 아닌 항목(문자열 등) 방어 필터
        segments = [s for s in segments if isinstance(s, dict)]
        if duration is None:
            duration = float(segments[-1].get("end", 0)) if segments else 0.0

        # Whisper 형식으로 변환 (avg_logprob=0.0 으로 교정 플래그 비활성)
        formatted = []
        for i, seg in enumerate(segments):
            formatted.append({
                "id":          i,
                "start":       float(seg.get("start", 0)),
                "end":         float(seg.get("end", 0)),
                "text":        seg.get("text", ""),
                "avg_logprob": 0.0,
            })

        print(f"     ✅ Gemini 전사 완료 ({len(formatted)}개 세그먼트 / {round(duration, 2)}초)")
        return {
            "video_info": {"path": video_path, "duration": round(duration, 2)},
            "segments":   formatted,
        }

    # ────────────────────────────────────────────────────
    # 🇰🇷 Clova NEST 전사 [1순위 — ASR A/B]
    # ────────────────────────────────────────────────────
    def transcribe_with_clova(self, video_path: str) -> dict:
        """
        Clova Speech(NEST) 장문 인식. 영상→오디오(wav) 추출 후 업로드.
        반환 형식은 Whisper/Gemini 전사와 동일 (video_info + segments).
        키/ffmpeg 미설정 시 RuntimeError → 호출부에서 Gemini 폴백.
        """
        if not (CLOVA_INVOKE_URL and CLOVA_SECRET_KEY):
            raise RuntimeError("Clova 자격증명(CLOVA_INVOKE_URL/SECRET_KEY) 미설정")

        import subprocess, requests, tempfile
        print(f"🇰🇷 Clova NEST 전사 시작: {video_path}")

        # 1) 영상 → 16kHz mono wav 추출
        wav_path = os.path.join(tempfile.gettempdir(), f"clova_{os.getpid()}.wav")
        cmd = [FFMPEG_BIN, "-y", "-i", video_path,
               "-ar", "16000", "-ac", "1", "-vn", wav_path]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            raise RuntimeError(f"ffmpeg 오디오 추출 실패: {e}")

        # 2) Clova Speech long-sentence 업로드 인식
        params = {
            "language": "ko-KR",
            "completion": "sync",
            "wordAlignment": True,
            "fullText": True,
        }
        headers = {"X-CLOVASPEECH-API-KEY": CLOVA_SECRET_KEY}
        try:
            with open(wav_path, "rb") as f:
                files = {
                    "media": f,
                    "params": (None, json.dumps(params), "application/json"),
                }
                resp = requests.post(CLOVA_INVOKE_URL, headers=headers,
                                     files=files, timeout=600)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Clova 인식 요청 실패: {e}")
        finally:
            try: os.remove(wav_path)
            except OSError: pass

        # 3) Whisper 형식으로 변환 (segments: ms → s)
        segments = []
        for i, seg in enumerate(data.get("segments", [])):
            segments.append({
                "id":          i,
                "start":       float(seg.get("start", 0)) / 1000.0,
                "end":         float(seg.get("end", 0)) / 1000.0,
                "text":        seg.get("text", ""),
                "avg_logprob": 0.0,
            })
        # segments가 비면 fullText라도 단일 세그먼트로
        if not segments and data.get("text"):
            segments = [{"id": 0, "start": 0.0, "end": 0.0,
                         "text": data["text"], "avg_logprob": 0.0}]
        duration = segments[-1]["end"] if segments else 0.0
        print(f"     ✅ Clova 전사 완료 ({len(segments)}개 세그먼트 / {round(duration,2)}초)")
        return {
            "video_info": {"path": video_path, "duration": round(duration, 2)},
            "segments":   segments,
        }

    def ab_test_asr(self, video_path: str, reference_text: str) -> dict:
        """
        같은 영상에 대해 Gemini vs Clova 전사를 돌려 CER을 비교.
        reference_text: 사람이 만든 정답 전사(골든). 낮은 CER이 우수.
        """
        out = {}
        try:
            g = self.transcribe_with_gemini(video_path)
            g_text = " ".join(s["text"] for s in g["segments"])
            out["gemini_cer"] = round(cer(reference_text, g_text), 4)
        except Exception as e:
            out["gemini_cer"] = None; out["gemini_error"] = str(e)
        try:
            c = self.transcribe_with_clova(video_path)
            c_text = " ".join(s["text"] for s in c["segments"])
            out["clova_cer"] = round(cer(reference_text, c_text), 4)
        except Exception as e:
            out["clova_cer"] = None; out["clova_error"] = str(e)
        if out.get("gemini_cer") is not None and out.get("clova_cer") is not None:
            out["winner"] = "clova" if out["clova_cer"] < out["gemini_cer"] else "gemini"
        return out

    # ────────────────────────────────────────────────────
    # 🧹 정규화
    # ────────────────────────────────────────────────────
    def normalize_transcript(self, raw_data: dict) -> dict:
        segments      = raw_data.get("segments", [])
        total_modified = 0
        for s in segments:
            original = s.get('text', '')
            cleaned  = normalize_text(original)
            if original != cleaned:
                s['text'] = cleaned
                total_modified += 1
        return {"segments": segments, "total_modified": total_modified,
                "video_info": raw_data.get("video_info", {})}

    # ────────────────────────────────────────────────────
    # 🤖 Gemini 교정 (배치 캐시 + 중간 재개)
    # ────────────────────────────────────────────────────
    def gemini_refine(
        self,
        segments:   list,
        batch_size: int  = 20,
        cache_dir:  str  = "",
        resume:     bool = True,
    ) -> list:
        """
        Gemini 교정 — 배치별 병렬 처리(_REFINE_CONCURRENCY 동시), 캐시 지원, _safe_json_parse 적용.
        """
        total_batches = (len(segments) + batch_size - 1) // batch_size
        refined_all   = [None] * total_batches

        use_cache = bool(cache_dir)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

        # ── 캐시된 배치 먼저 로드 ─────────────────────────────
        pending_idxs = []
        for idx in range(total_batches):
            cp = os.path.join(cache_dir, f"_refined_batch_{idx}.json") if use_cache else ""
            if resume and use_cache and os.path.exists(cp):
                with open(cp, 'r', encoding='utf-8') as f:
                    refined_all[idx] = json.load(f)
                print(f"  📦 배치 {idx+1}/{total_batches} — 캐시 로드 (건너뜀)")
            else:
                pending_idxs.append(idx)

        if not pending_idxs:
            return [seg for batch in refined_all if batch for seg in batch]

        print(f"  📝 Gemini 교정 — {len(pending_idxs)}개 배치 병렬 실행 (동시 최대 {_REFINE_CONCURRENCY}개)")
        sem = threading.Semaphore(_REFINE_CONCURRENCY)

        def _do_batch(idx: int) -> tuple:
            batch = segments[idx * batch_size: (idx + 1) * batch_size]
            cp    = os.path.join(cache_dir, f"_refined_batch_{idx}.json") if use_cache else ""
            flagged = [
                {**s, "_low_confidence": True} if s.get("avg_logprob", 0) < -0.8 else dict(s)
                for s in batch
            ]
            prompt = (
                f"너는 한국어 STT 교정 전문가야.\n"
                f"[교정 규칙]\n"
                f"1. '_low_confidence': true 항목은 꼼꼼히 교정.\n"
                f"2. 발음이 유사하지만 맥락상 틀린 단어를 바로잡아 (예: '뒷깡고'→'뒷광고').\n"
                f"3. start/end 절대 변경 금지. 내용 추가 금지, 교정만.\n"
                f"[데이터]\n{json.dumps(flagged, ensure_ascii=False)}\n"
                f"[출력 형식 — JSON 배열만, 설명 없이]\n"
                f'[{{"segment_id":1,"start":0.0,"end":3.5,"original_text":"원본","corrected_text":"교정문","correction_reason":""}}]'
            )
            with sem:
                for attempt in range(3):
                    try:
                        resp = self.client.models.generate_content(
                            model=GEMINI_REFINE_MODEL,
                            contents=prompt,
                            config={
                                "system_instruction": self.system_instruction,
                                "response_mime_type": "application/json",
                                "temperature": 0.1,
                            },
                        )
                        self.cost.add(resp, "refine")
                        batch_result = _safe_json_parse(resp.text, f"refine batch {idx+1}")
                        if not batch_result:
                            raise ValueError("JSON 파싱 실패")
                        id_map = {item.get("segment_id"): item for item in batch_result}
                        merged = []
                        for s in batch:
                            sid     = s.get("segment_id", s.get("id"))
                            refined = id_map.get(sid, {})
                            merged.append({
                                "segment_id":        sid,
                                "start":             s["start"],
                                "end":               s["end"],
                                "original_text":     s.get("text", ""),
                                "corrected_text":    refined.get("corrected_text", s.get("text", "")),
                                "correction_reason": refined.get("correction_reason", ""),
                            })
                        if use_cache and cp:
                            with open(cp, 'w', encoding='utf-8') as f:
                                json.dump(merged, f, indent=2, ensure_ascii=False)
                            print(f"       💾 배치 {idx+1} 캐시 저장 → {cp}")
                        print(f"       ✅ 배치 {idx+1}/{total_batches} 완료 ({len(merged)}개)")
                        return idx, merged
                    except Exception as e:
                        if "429" in str(e):
                            wait = (attempt + 1) * 15 + idx * 2
                            print(f"  ⚠️  배치 {idx+1} 할당량 초과 → {wait}초 대기 ({attempt+1}/3)")
                            time.sleep(wait)
                        else:
                            print(f"  ⚠️  배치 {idx+1} 오류 ({attempt+1}/3): {e}")
                            time.sleep(2)
            print(f"  ⛔ 배치 {idx+1} 최종 실패 — 원본 유지")
            return idx, [
                {
                    "segment_id":        s.get("segment_id", s.get("id")),
                    "start":             s["start"],
                    "end":               s["end"],
                    "original_text":     s.get("text", ""),
                    "corrected_text":    s.get("text", ""),
                    "correction_reason": "API 오류",
                }
                for s in batch
            ]

        with ThreadPoolExecutor(max_workers=min(len(pending_idxs), _REFINE_CONCURRENCY)) as executor:
            futures = {executor.submit(_do_batch, idx): idx for idx in pending_idxs}
            for fut in as_completed(futures):
                idx, result = fut.result()
                refined_all[idx] = result

        return [seg for batch in refined_all if batch for seg in batch]

    # ────────────────────────────────────────────────────
    # 🚀 전사 파이프라인
    # ────────────────────────────────────────────────────
    def run_whisper_pipeline(
        self,
        video_path:   str,
        output_dir:   str  = "samples/transcripts",
        gemini_batch: int  = 10,
        use_cache:    bool = True,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]

        raw_path     = os.path.join(output_dir, f"{base}_raw.json")
        refined_path = os.path.join(output_dir, f"{base}_refined.json")
        batch_dir    = os.path.join(output_dir, f"{base}_batch_cache")

        engine_label = "Gemini" if TRANSCRIBE_ENGINE == "gemini" else "Whisper"
        if use_cache and os.path.exists(raw_path):
            print(f"📦 [1/3] {engine_label} 전사 캐시 발견 → {raw_path} (건너뜀)")
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        else:
            if TRANSCRIBE_ENGINE == "gemini":
                print("🤖 [1/3] Gemini 전사 시작...")
                raw = self.transcribe_with_gemini(video_path)
            else:
                print("🎙️  [1/3] Whisper 전사 시작...")
                raw = self.transcribe(video_path)
            _save_json(raw, raw_path)
            print(f"     ✅ 전사 완료 → {raw_path}  ({len(raw['segments'])}개 / {raw['video_info']['duration']}초)")

        norm = self.normalize_transcript(raw)
        print(f"✅ [2/3] 정규화 완료  (수정: {norm['total_modified']}개)")

        if use_cache and os.path.exists(refined_path):
            print(f"📦 [3/3] Gemini 교정 캐시 발견 → {refined_path} (건너뜀)")
            with open(refined_path, 'r', encoding='utf-8') as f:
                refined_data = json.load(f)
        else:
            print(f"🤖 [3/3] Gemini 교정 시작... (배치 크기: {gemini_batch})")
            refined_list = self.gemini_refine(
                norm['segments'],
                batch_size = gemini_batch,
                cache_dir  = batch_dir,
                resume     = True,
            )
            total_corrected = sum(
                1 for s in refined_list if s.get("corrected_text") != s.get("original_text")
            )
            refined_data = {
                "segments":        refined_list,
                "video_info":      raw['video_info'],
                "total_corrected": total_corrected,
            }
            _save_json(refined_data, refined_path)
            print(f"     ✅ Gemini 교정 완료 → {refined_path}  (교정: {total_corrected}개)")

        return {
            "base":         base,
            "raw_path":     raw_path,
            "refined_path": refined_path,
            "raw_data":     raw,
            "refined_data": refined_data,
        }

    # ────────────────────────────────────────────────────
    # 🏎️ ML 로컬 분류 (controversy_model.joblib)
    # ────────────────────────────────────────────────────
    def _ml_classify(self, text: str) -> dict:
        """단일 텍스트를 ML 모델로 즉시 분류 → classify_controversy 형식 반환"""
        probs   = self.ml_model.predict_proba([text])[0]   # (n_classes,)
        classes = list(self.ml_mlb.classes_)               # ['L01', ..., 'L12']

        primary_idx = int(np.argmax(probs))
        primary_id  = classes[primary_idx]

        threshold = 0.3
        selected  = [classes[i] for i, p in enumerate(probs) if p >= threshold]
        if not selected:
            selected = [primary_id]

        labels_full = [f"{lid} {_LABEL_DESCS.get(lid, '')}" for lid in selected]
        return {
            "labels":  labels_full,
            "primary": primary_id,
            "reason":  f"ML 분류 (신뢰도 {probs[primary_idx]:.2f})",
        }

    def _ml_classify_segments(self, segments: list) -> list:
        """세그먼트 배치를 ML 모델로 즉시 분류 → analyze_transcript_segments 형식 반환"""
        if not segments:
            return []

        texts = [
            normalize_text(s.get("corrected_text") or s.get("text", "")) or "."
            for s in segments
        ]

        probs_mat    = self.ml_model.predict_proba(texts)        # (n, n_classes)
        preds_mat    = self.ml_model.predict(texts)              # (n, n_classes) binarized
        decoded_list = self.ml_mlb.inverse_transform(preds_mat) # list of tuples
        classes      = list(self.ml_mlb.classes_)

        results = []
        for i, seg in enumerate(segments):
            decoded  = decoded_list[i]   # e.g. ('L01', 'L03')
            prob_row = probs_mat[i]

            if not decoded:
                primary_id = "L12"
                conf       = 0.0
            else:
                primary_id = max(
                    decoded,
                    key=lambda lid: prob_row[classes.index(lid)] if lid in classes else 0.0,
                )
                conf = prob_row[classes.index(primary_id)] if primary_id in classes else 0.0

            label_str = f"{primary_id} {_LABEL_DESCS.get(primary_id, '해당 없음')}"
            results.append({
                "segment_id":     seg.get("segment_id", seg.get("id", i)),
                "start":          seg.get("start"),
                "end":            seg.get("end"),
                "timestamp":      _seg_timestamp(seg),   # 'mm:ss–mm:ss'
                "corrected_text": seg.get("corrected_text") or seg.get("text", ""),
                "label":          label_str,
                "reason":         f"ML 분류 (신뢰도 {conf:.2f})",
            })
        return results

    # ────────────────────────────────────────────────────
    # 🎯 2-Tier 분류 [2순위 — 핵심 구조 개선]
    #   Tier1(SVM): bag-of-words라 맥락·반어·인용·부정문을 못 읽음.
    #               → '판정'이 아니라 '의심 구간 트리거'로만 사용(재현율 우선).
    #   Tier2(LLM): 트리거 구간 + 앞뒤 맥락 + 유사사례 + 룰매칭을 함께 보고 최종 판정.
    # ────────────────────────────────────────────────────
    def _tier1_trigger(self, segments: list) -> list:
        """SVM을 낮은 임계값으로 돌려 '의심 세그먼트' 인덱스/라벨을 넓게 뽑음(재현율↑)."""
        if self.ml_model is None or not segments:
            return []
        texts = [
            normalize_text(s.get("corrected_text") or s.get("text", "")) or "."
            for s in segments
        ]
        probs_mat = self.ml_model.predict_proba(texts)
        classes   = list(self.ml_mlb.classes_)
        triggered = []
        for i, prob_row in enumerate(probs_mat):
            hits = [
                {"label": classes[j], "prob": float(p)}
                for j, p in enumerate(prob_row)
                if classes[j] != "L12" and p >= TIER1_TRIGGER_THRESHOLD
            ]
            if not hits:
                continue
            hits.sort(key=lambda x: -x["prob"])
            high_sev = any(h["label"] in HIGH_SEVERITY_LABELS for h in hits)
            triggered.append({"idx": i, "hits": hits, "high_severity": high_sev})
        return triggered

    def _tier2_adjudicate(self, segments, trig, rule_hits, similar_titles) -> dict:
        """
        트리거된 한 구간을 맥락과 함께 LLM에 보내 최종 판정(JSON).
        앞뒤 TIER2_CONTEXT_WINDOW개 세그먼트를 동봉해 반어·인용·부정문을 구분.
        """
        i = trig["idx"]
        lo = max(0, i - TIER2_CONTEXT_WINDOW)
        hi = min(len(segments), i + TIER2_CONTEXT_WINDOW + 1)
        ctx_lines = []
        for j in range(lo, hi):
            mark = "👉" if j == i else "  "
            txt  = segments[j].get("corrected_text") or segments[j].get("text", "")
            ctx_lines.append(f"{mark} [{j}] {txt}")
        context = "\n".join(ctx_lines)
        tier1_labels = ", ".join(
            f"{h['label']}({h['prob']:.2f})" for h in trig["hits"]
        )

        prompt = (
            "너는 한국 캔슬컬처 리스크 판정관이다. 1차 필터(SVM)가 아래 '👉' 구간을 "
            "의심으로 표시했다. 하지만 SVM은 맥락을 못 읽으니, 네가 앞뒤 맥락을 보고 "
            "'실제 논란 소지'인지 최종 판정하라.\n"
            "[중요] 반어·인용·부정문(\"이건 ~라는 뜻이 절대 아니다\")이면 is_controversy=false.\n\n"
            f"[1차 의심 라벨] {tier1_labels}\n"
            f"[룰 매칭] {rule_hits if rule_hits else '없음'}\n"
            f"[유사 과거사례] {', '.join(similar_titles) if similar_titles else '없음'}\n\n"
            f"[맥락 (👉가 판정 대상)]\n{context}\n\n"
            f"[라벨 목록] {LABEL_INFO}\n\n"
            "[출력 — JSON만]\n"
            '{"is_controversy":true,"category":"L03","severity":"DANGER",'
            '"rationale":"앞뒤 맥락상 혐오 표현을 진심으로 사용","quote":"문제 문장"}'
        )
        try:
            resp = self.client.models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.1},
            )
            self.cost.add(resp, "tier2")
            res = _safe_json_parse(resp.text, f"Tier2 seg {i}") or {}
        except Exception as e:
            res = {"is_controversy": True, "category": trig["hits"][0]["label"],
                   "severity": "ALERT", "rationale": f"Tier2 실패, Tier1 유지: {e}", "quote": ""}
        res["segment_idx"]  = i
        res["tier1_labels"] = [h["label"] for h in trig["hits"]]
        res["start"]        = segments[i].get("start")
        res["end"]          = segments[i].get("end")
        res["timestamp"]    = _seg_timestamp(segments[i])   # 'mm:ss–mm:ss'
        return res

    def two_tier_classify(self, segments: list, rule_hits=None, similar_titles=None) -> dict:
        """
        전체 오케스트레이션:
          1) Tier1 SVM 트리거(재현율 우선)
          2) 심각도 라우팅 — 고심각도 라벨은 무조건 Tier2, 경미 구간은 비용 상한 내에서
          3) Tier2 LLM 맥락 판정(병렬)
        반환: {triggered_count, adjudicated:[...], confirmed:[...], tier1_only:[...]}
        """
        rule_hits      = rule_hits or []
        similar_titles = similar_titles or []
        triggered = self._tier1_trigger(segments)
        if not triggered:
            return {"triggered_count": 0, "adjudicated": [], "confirmed": [], "tier1_only": []}

        if not ENABLE_TIER2:
            return {"triggered_count": len(triggered), "adjudicated": [],
                    "confirmed": [], "tier1_only": triggered}

        # 심각도 라우팅: 고심각도 먼저, 그다음 일반 — 비용 상한까지만 Tier2
        triggered.sort(key=lambda t: (not t["high_severity"], -t["hits"][0]["prob"]))
        to_judge   = triggered[:TIER2_MAX_SEGMENTS]
        tier1_only = triggered[TIER2_MAX_SEGMENTS:]

        adjudicated = []
        with ThreadPoolExecutor(max_workers=_REFINE_CONCURRENCY) as ex:
            futs = {
                ex.submit(self._tier2_adjudicate, segments, t, rule_hits, similar_titles): t
                for t in to_judge
            }
            for fut in as_completed(futs):
                try:
                    adjudicated.append(fut.result())
                except Exception as e:
                    print(f"   ⚠️ Tier2 판정 실패: {e}")

        confirmed = [a for a in adjudicated if a.get("is_controversy")]
        adjudicated.sort(key=lambda a: a.get("segment_idx", 0))
        confirmed.sort(key=lambda a: a.get("segment_idx", 0))
        print(f"   🎯 2-Tier: 트리거 {len(triggered)} → Tier2 판정 {len(adjudicated)} "
              f"→ 확정 논란 {len(confirmed)} (오탐 {len(adjudicated)-len(confirmed)} 제거)")
        return {
            "triggered_count": len(triggered),
            "adjudicated":     adjudicated,
            "confirmed":       confirmed,
            "tier1_only":      [{"idx": t["idx"], "hits": t["hits"]} for t in tier1_only],
        }

    # ────────────────────────────────────────────────────
    # 🔍 RAG 검색 + 패턴 분석
    # ────────────────────────────────────────────────────
    def search_and_analyze(self, query: str, k: int = 3) -> tuple:
        resp = self.client.models.embed_content(model=EMBED_MODEL, contents=query)
        qe   = np.array([resp.embeddings[0].values]).astype('float32')
        dists, idxs = self.index.search(qe, k)
        top = [self.cases[i] for i in idxs[0] if i != -1]

        ctx = "".join(
            f"사례 {i}: {c['title']}\n- 유형: {c.get('controversy_type','N/A')}\n"
            f"- 상황: {c['summary']}\n- 대응: {c.get('response_pattern',[])}\n"
            f"- 결과: {c.get('keyframes',[])}\n\n"
            for i, c in enumerate(top, 1)
        )
        prompt = (
            f"입력된 사건: \"{query}\"\n\n유사 사례:\n{ctx}\n"
            "다음 3가지를 요약해 주세요:\n"
            "1. 위기 확산의 공통적 경로\n"
            "2. 과거 대응 방식에 따른 여론의 반응 패턴\n"
            "3. 위기가 심화되었던 결정적 트리거(Trigger)"
        )
        resp2 = self.client.models.generate_content(
            model=GEN_MODEL, contents=prompt,
            config={"system_instruction": self.system_instruction},
        )
        self.cost.add(resp2, "search")
        return top, dists[0], resp2.text

    # ────────────────────────────────────────────────────
    # 🏷️ 논란 유형 분류 (ML 우선 → Gemini 폴백)
    # ────────────────────────────────────────────────────
    def classify_controversy(self, query: str) -> dict:
        if self.ml_model is not None:
            return self._ml_classify(query)

        # ── Gemini 폴백 ───────────────────────────────────
        prompt = (
            f"아래 사건을 읽고 논란 유형 라벨을 골라줘.\n\n사건: \"{query}\"\n\n"
            f"라벨 목록: {self.label_prompt_str}\n\n"
            "JSON 형식으로만 답해줘:\n"
            '{"labels":["L03 혐오 표현"],"primary":"L03","reason":"이유"}'
        )
        resp = self.client.models.generate_content(
            model=GEN_MODEL, contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        result = _safe_json_parse(resp.text, "논란 유형 분류")
        return result if result else {"labels": [], "primary": "Unknown", "reason": resp.text.strip()}

    # ────────────────────────────────────────────────────
    # 📝 자막 세그먼트 논란 분류 (ML 우선 → Gemini 폴백)
    # ────────────────────────────────────────────────────
    def analyze_transcript_segments(self, segments: list) -> list:
        if self.ml_model is not None:
            return self._ml_classify_segments(segments)

        # ── Gemini 폴백 ───────────────────────────────────
        clean_segs = []
        for s in segments:
            clean_segs.append({
                "segment_id": s.get("segment_id", s.get("id")),
                "text":       normalize_text(s.get("corrected_text") or s.get("text", "")),
            })
        prompt = (
            f"유튜브 논란 분석 전문가로서 아래 교정된 자막을 분류해줘.\n\n"
            f"[논란 유형]\n{self.label_prompt_str}\n\n"
            f"[데이터]\n{json.dumps(clean_segs, ensure_ascii=False)}\n\n"
            "[결과 포맷] JSON 배열만:\n"
            '[{"segment_id":1,"corrected_text":"교정문","label":"유형","reason":"이유"}]'
        )
        resp = self.client.models.generate_content(
            model=GEN_MODEL, contents=prompt,
            config={
                "system_instruction": self.system_instruction,
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )
        result = _safe_json_parse(resp.text, "자막 세그먼트 분류")
        return result if result else []

    # ────────────────────────────────────────────────────
    # 🔎 [L04 Stage1] 검증 필요 주장 추출 (진위 판정 X)
    # ────────────────────────────────────────────────────
    def extract_checkworthy_claims(self, segments: list) -> dict:
        """
        영상에서 '검증이 필요한 사실 주장(check-worthy claim)'만 추출한다.
          · 진위(참/거짓)는 판정하지 않는다 — '검증 대상'만 골라 타임스탬프/도메인 태깅.
          · 번호 매긴 세그먼트를 LLM에 주고 각 주장의 segment_idx 를 돌려받아
            타임스탬프를 정확히 역추적한다(인용문 문자열 매칭보다 견고).
        반환: {"claims": [...], "count": n}
        """
        if not segments:
            return {"claims": [], "count": 0}

        lines = []
        for i, s in enumerate(segments):
            txt = (s.get("corrected_text") or s.get("text", "")).strip()
            if txt:
                lines.append(f"[{i}] {txt}")
        if not lines:
            return {"claims": [], "count": 0}
        numbered = "\n".join(lines)

        prompt = (
            "너는 한국 유튜브 영상의 '사실 주장'을 식별하는 팩트체크 보조자다.\n"
            "아래 번호가 매겨진 자막에서 '검증이 필요한 사실 주장'만 골라라.\n\n"
            "[검증 대상 = 사실 주장]\n"
            "- 객관적으로 참/거짓을 가릴 수 있는 진술(통계·수치, 의학/건강, 금융/투자, "
            "역사적 사실, 과학, 출처·인용 등).\n"
            "[제외 = 검증 대상 아님]\n"
            "- 개인 의견·감상·취향('내 생각엔', '맛있다'), 농담, 인사말, 단순 잡담.\n\n"
            "[매우 중요] 진위(참/거짓)는 절대 판정하지 마라. '검증이 필요한지'만 표시한다.\n"
            "- domain 은 반드시: 건강/의학, 금융/투자, 통계/수치, 역사/사회, 과학/기술, 인용/출처, 기타 중 하나.\n"
            "- checkworthiness 는 HIGH/MEDIUM/LOW (피해 가능성 큰 건강·금융 주장일수록 HIGH).\n"
            "- segment_idx 는 그 주장이 나온 자막 줄의 [번호].\n\n"
            "[출력 — JSON만, 설명·마크다운 없이]\n"
            '{"claims":[{"segment_idx":0,"claim":"주장 원문 인용","domain":"건강/의학",'
            '"checkworthiness":"HIGH","reason":"왜 검증이 필요한지 한 줄"}]}\n\n'
            f"[자막]\n{numbered}"
        )

        try:
            resp = self.client.models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                    "max_output_tokens": 4096,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            self.cost.add(resp, "claim_stage1")
            data = _safe_json_parse(resp.text, "검증대상 주장 추출")
        except Exception as e:
            print(f"   ⚠️ 검증대상 주장 추출 실패: {e}")
            return {"claims": [], "count": 0}

        raw_claims = data.get("claims", []) if isinstance(data, dict) else []
        VALID_DOMAINS = {"건강/의학", "금융/투자", "통계/수치", "역사/사회",
                         "과학/기술", "인용/출처", "기타"}
        VALID_CW = {"HIGH", "MEDIUM", "LOW"}
        out = []
        for c in raw_claims:
            if not isinstance(c, dict):
                continue
            claim = (c.get("claim") or "").strip()
            if not claim:
                continue
            try:
                idx = int(c.get("segment_idx"))
            except (TypeError, ValueError):
                idx = None
            ts = "—"
            if idx is not None and 0 <= idx < len(segments):
                ts = _seg_timestamp(segments[idx])
            else:
                idx = None
            domain = c.get("domain") if c.get("domain") in VALID_DOMAINS else "기타"
            cw = (c.get("checkworthiness") or "").upper()
            cw = cw if cw in VALID_CW else "MEDIUM"
            out.append({
                "segment_idx":     idx,
                "timestamp":       ts,
                "claim":           claim,
                "domain":          domain,
                "checkworthiness": cw,
                "reason":          (c.get("reason") or "").strip(),
            })

        # 검증가치 높은 순 → 등장 순
        _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        out.sort(key=lambda x: (_order.get(x["checkworthiness"], 1),
                                x["segment_idx"] if x["segment_idx"] is not None else 10**9))
        print(f"   🔎 검증 필요 주장 {len(out)}건 추출 (Stage1 · 진위판정 없음)")
        return {"claims": out, "count": len(out)}

    # ────────────────────────────────────────────────────
    # 📝 사건 개요 요약
    # ────────────────────────────────────────────────────
    def summarize_incident(self, refined_segments, youtube_meta: dict = None) -> dict:
        """자막 + 유튜브 메타데이터를 함께 활용해 사건 개요 생성"""
        full_text = " ".join([
            s.get('corrected_text') or s.get('text', '')
            for s in refined_segments
        ]).strip()

        if not full_text or len(full_text) < 10:
            return {
                "summary_title":    "영상 내용 분석 결과",
                "incident_overview": "전사 데이터를 기반으로 분석되었습니다.",
            }

        if len(full_text) > 4000:
            full_text = full_text[:2000] + " (중략) " + full_text[-2000:]

        # 유튜브 메타 정보 추가 (있을 경우)
        meta_hint = ""
        if youtube_meta:
            meta_hint = (
                f"\n\n[유튜브 영상 정보]\n"
                f"제목: {youtube_meta.get('title','')}\n"
                f"채널: {youtube_meta.get('channel','')} (구독자 {youtube_meta.get('subscriber_count',0):,}명)\n"
                f"조회수: {youtube_meta.get('view_count',0):,} | 좋아요: {youtube_meta.get('like_count',0):,} | 댓글: {youtube_meta.get('comment_count',0):,}\n"
                f"업로드: {youtube_meta.get('upload_date','')}"
            )

        prompt = f"""당신은 유튜브 영상 내용을 요약하는 어시스턴트입니다.
아래 영상 전사(STT) 텍스트{' 및 유튜브 메타 정보' if meta_hint else ''}를 읽고, 지시한 형식에 맞게 JSON으로만 응답하세요.
절대로 전사 텍스트를 그대로 복사하지 마세요. 반드시 자신의 말로 요약하세요.

[전사 텍스트]
{full_text}{meta_hint}

[출력 규칙]
- summary_title: 영상의 핵심 내용을 30자 내외 명사형으로 작성.
- incident_overview: 화자가 어떤 콘텐츠에서, 어떤 행동을 했고, 어떤 논란 요소가 있었는지 500자 내외로 상세히 서술.

[응답 형식 — JSON만]
{{"summary_title": "...", "incident_overview": "..."}}"""

        try:
            response = self.client.models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config={'response_mime_type': 'application/json', 'temperature': 0.2},
            )
            result = _safe_json_parse(response.text, "사건 개요 요약")
            if result is None:
                raise ValueError("사건 개요 JSON 파싱 실패")
            title    = str(result.get("summary_title", "")).strip()
            overview = str(result.get("incident_overview", "")).strip()
            if not title or title in {"...", "summary_title"}:
                title = "영상 내용 분석 결과"
            if not overview or overview in {"...", "incident_overview"}:
                overview = "전사 데이터를 기반으로 분석되었습니다."
            if len(title)    > 60:   title    = title[:60].rstrip()    + "…"
            if len(overview) > 1000: overview = overview[:1000].rstrip() + "…"
            return {"summary_title": title, "incident_overview": overview}
        except Exception as e:
            print(f"⚠️ 사건 개요 생성 실패: {e}")
            return {
                "summary_title":    "영상 내용 분석 결과",
                "incident_overview": "전사 데이터를 기반으로 분석되었습니다.",
            }

    # ────────────────────────────────────────────────────
    # 🎞️ 키프레임 분석
    # ────────────────────────────────────────────────────
    def analyze_video_frames(self, video_path: str) -> list:
        """키프레임 3개를 추출한 뒤 Gemini 호출을 병렬로 실행해 분석 시간 단축"""
        print(f"📹 키프레임 분석 중: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── 프레임 추출 (순차, I/O bound) ──────────────────
        frame_images = {}
        for i in range(3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * (i + 1) / 4))
            ret, frame = cap.read()
            if not ret:
                continue
            img_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(img_path, frame)
            frame_images[i] = img_path
        cap.release()

        if not frame_images:
            return []

        # ── Gemini 호출 병렬 실행 ──────────────────────────
        def _analyze_one(idx: int, img_path: str) -> dict | None:
            try:
                with PIL.Image.open(img_path) as img:
                    resp = self.client.models.generate_content(
                        model=GEN_MODEL,
                        contents=[
                            "이 영상 캡처에서 보이는 객관적 요소(자막, 행동, 구도)를 "
                            "5자 이내로 요약해줘. 감정이나 위험도는 판단하지 마.", img,
                        ],
                    )
                    self.cost.add(resp, "frame")
                    out = {"idx": idx, "time": f"Point {idx+1}", "tag": resp.text.strip()}
                    # [4순위] 화면 텍스트 OCR — 자막 욕설/혐오 표현은 음성에 없어도 잡아야 함
                    ocr_text = _ocr_screen_text(img_path)
                    if ocr_text:
                        out["screen_text"] = ocr_text
                    # [4순위] 한국 특화 시각 트리거(집게손 등 밈/제스처) — 확장 포인트
                    memes = _detect_korea_visual_triggers(img_path)
                    if memes:
                        out["visual_triggers"] = memes
                    return out
            except Exception as e:
                print(f"⚠️ 프레임 {idx+1} 분석 실패: {e}")
                return None
            finally:
                try:
                    os.remove(img_path)
                except:
                    pass

        frames = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_analyze_one, idx, path): idx
                for idx, path in frame_images.items()
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    frames.append(result)

        # 원래 순서대로 정렬
        frames.sort(key=lambda x: x["idx"])
        for f in frames:
            f.pop("idx", None)
        return frames

    def rule_scan(self, text: str) -> dict:
        return rule_engine(normalize_text(text), self.rules)

    def rule_scan_union(self, *texts: str) -> dict:
        """
        raw + refined + 화면 OCR 텍스트를 모두 스캔해 매칭 합집합(union).
        - raw만 스캔하면 ASR 오타로 키워드를 놓치고,
        - refined만 하면 교정 중 표현이 순화돼 놓칠 수 있어 양쪽 다 본다.
        rule_engine은 '첫 매칭 1건'만 반환하므로, 여기서 모든 정책·키워드를 직접 훑어
        전체 매칭을 모은다. 하위호환을 위해 가장 심각한 매칭을 최상위 필드로 둔다.
        """
        _sev_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "UNKNOWN": 0}
        norm_texts = [normalize_text(t) for t in texts if t]
        matches, seen = [], set()
        for pname, detail in self.rules.items():
            if not isinstance(detail, dict):
                continue
            for word in detail.get("keywords", []):
                if any(word in nt for nt in norm_texts):
                    key = (pname, word)
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append({
                        "policy":       pname,
                        "action":       detail.get("action", "REVIEW"),
                        "severity":     detail.get("severity", "UNKNOWN"),
                        "matched_word": word,
                    })
        if not matches:
            return {"hit": False, "all_matches": []}
        matches.sort(key=lambda m: -_sev_order.get(m["severity"], 0))
        top = dict(matches[0])           # 하위호환: 단일 매칭 형태 유지
        top["hit"]         = True
        top["all_matches"] = matches     # 신규: 전체 매칭 목록
        return top

    def get_risk_guide(self, controversy_type: str, spread_stage: str) -> list:
        return self.risk_consultant.polish_tone(
            self.risk_consultant.get_worst_actions(controversy_type, spread_stage)
        )

    # ────────────────────────────────────────────────────
    # 🎬 영상 전체 분석 (유튜브 URL 또는 로컬 파일 모두 지원)
    # ────────────────────────────────────────────────────
    def analyze_video_full(
        self,
        video_input:  str,          # 유튜브 URL / Google Drive URL / 로컬 파일 경로
        query:        str  = "",
        output_dir:   str  = "samples/transcripts",
        download_dir: str  = "downloads",
        gemini_batch: int  = 20,
        use_cache:    bool = True,
    ) -> tuple:
        t0 = time.time()
        youtube_meta = None

        # ── URL 감지: 유튜브 → 다운로드+메타, Google Drive → 다운로드, 로컬 → 그대로 ─────
        if is_youtube_url(video_input):
            print(f"\n{'='*63}")
            print(f"🔗 유튜브 URL 감지: {video_input}")
            print(f"{'='*63}")
            youtube_meta = download_youtube_video(video_input, download_dir)
            video_path   = youtube_meta['video_path']
        elif is_google_drive_url(video_input):
            print(f"\n{'='*63}")
            print(f"📁 Google Drive URL 감지: {video_input}")
            print(f"{'='*63}")
            _gdrive_dl = download_google_drive_video(video_input, download_dir)
            video_path = _gdrive_dl['video_path']
        else:
            video_path = video_input

        print(f"\n{'='*63}")
        print(f"🎬 영상 분석 시작: {os.path.basename(video_path)}")
        print(f"{'='*63}")

        base = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        raw_path     = os.path.join(output_dir, f"{base}_raw.json")
        refined_path = os.path.join(output_dir, f"{base}_refined.json")
        batch_dir    = os.path.join(output_dir, f"{base}_batch_cache")

        # ── [1/3] 전사 ────────────────────────────────────
        engine_label = {"gemini": "Gemini", "clova": "Clova NEST"}.get(TRANSCRIBE_ENGINE, "Whisper")
        if use_cache and os.path.exists(raw_path):
            print(f"📦 [1/3] {engine_label} 전사 캐시 발견 → {raw_path} (건너뜀)")
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        else:
            if TRANSCRIBE_ENGINE == "clova":
                try:
                    print("🇰🇷 [1/3] Clova NEST 전사 시작...")
                    raw = self.transcribe_with_clova(video_path)
                except Exception as e:
                    print(f"⚠️ Clova 전사 실패({e}) → Gemini 폴백")
                    raw = self.transcribe_with_gemini(video_path)
            elif TRANSCRIBE_ENGINE == "gemini":
                print("🤖 [1/3] Gemini 전사 시작...")
                raw = self.transcribe_with_gemini(video_path)
            else:
                print("🎙️  [1/3] Whisper 전사 시작...")
                raw = self.transcribe(video_path)
            _save_json(raw, raw_path)
            print(f"     ✅ 전사 완료 → {raw_path}  ({len(raw['segments'])}개 / {raw['video_info']['duration']}초)")

        # ── [2/3] 정규화 + 즉시 처리 ─────────────────────
        norm = self.normalize_transcript(raw)
        print(f"✅ [2/3] 정규화 완료  (수정: {norm['total_modified']}개)")

        raw_query           = " ".join(s.get("text", "") for s in norm['segments'])[:1000]
        raw_transcript_text = " ".join(s.get("text", "") for s in norm['segments'])[:3000]

        # [2순위] Tier1 SVM 신호를 NATAM 입력으로 명시 연결 — 의심 라벨을 평가 컨텍스트로 제공
        tier1_trig   = self._tier1_trigger(norm['segments'])
        tier1_counts = {}
        for t in tier1_trig:
            for h in t["hits"]:
                tier1_counts[h["label"]] = tier1_counts.get(h["label"], 0) + 1
        tier1_signal = ", ".join(
            f"{_LABEL_DESCS.get(k, k)}×{v}" for k, v in
            sorted(tier1_counts.items(), key=lambda x: -x[1])
        ) or "특이 신호 없음"
        natam_summary_input = f"[1차 분류기 의심 신호] {tier1_signal}\n{raw_query[:500]}"

        rule_result = self.rule_scan(raw_query)   # 초기 스캔(아래에서 refined+OCR로 union 갱신)

        # ── 배경 실행: 교정과 동시에 NATAM·유사사례·키프레임 처리 ──
        print("⏳ 배경 분석 시작 (NATAM · 유사사례 · 키프레임) — [3/3] 교정과 병렬 실행")
        bg_exec    = ThreadPoolExecutor(max_workers=3)
        fut_natam  = bg_exec.submit(
            assess_natam_risk,
            self.client, self.system_instruction,
            raw_transcript_text, natam_summary_input, GEN_MODEL,
        )
        fut_search = bg_exec.submit(self.search_and_analyze, raw_query)
        fut_frames = bg_exec.submit(self.analyze_video_frames, video_path)

        # ── [3/3] Gemini 교정 (배치 병렬) ────────────────
        if use_cache and os.path.exists(refined_path):
            print(f"📦 [3/3] Gemini 교정 캐시 발견 → {refined_path} (건너뜀)")
            with open(refined_path, 'r', encoding='utf-8') as f:
                refined_data = json.load(f)
        else:
            print(f"🤖 [3/3] Gemini 교정 시작... (배치: {gemini_batch}개씩, 최대 {_REFINE_CONCURRENCY}개 동시)")
            refined_list = self.gemini_refine(
                norm['segments'],
                batch_size = gemini_batch,
                cache_dir  = batch_dir,
                resume     = True,
            )
            total_corrected = sum(
                1 for s in refined_list
                if s.get("corrected_text") != s.get("original_text")
            )
            refined_data = {
                "segments":        refined_list,
                "video_info":      raw['video_info'],
                "total_corrected": total_corrected,
            }
            _save_json(refined_data, refined_path)
            print(f"     ✅ Gemini 교정 완료 → {refined_path}  (교정: {total_corrected}개)")

        refined_segments = refined_data['segments']

        # ── ML 즉시 처리 (로컬, Gemini 호출 없음) ────────
        if not query:
            query = " ".join(
                s.get("corrected_text") or s.get("text", "")
                for s in refined_segments
            )[:1000]
        transcript_text = " ".join(
            s.get("corrected_text") or s.get("text", "")
            for s in refined_segments
        )[:3000]

        print("\n⚡ ML 분류 즉시 처리 중...")
        transcript_analysis = self.analyze_transcript_segments(refined_segments)
        classification      = self.classify_controversy(query)
        print(f"   ✅ transcript 완료 ({len(transcript_analysis)}개 세그먼트)")
        print(f"   ✅ classify 완료 → primary: {classification.get('primary','?')}")

        # ── 요약은 refined 텍스트로 (품질 우선, 교정 완료 후 시작) ──
        fut_summary = bg_exec.submit(self.summarize_incident, refined_segments, youtube_meta)
        # ── [L04 Stage1] 검증 필요 주장 추출 (교정문 기반, 병렬) ──
        fut_claims  = bg_exec.submit(self.extract_checkworthy_claims, refined_segments)

        # ── 배경 분석 결과 수집 ───────────────────────────
        print("⏳ 배경 분석 결과 수집 중...")

        def _get(fut, default, label):
            try:
                r = fut.result(timeout=90)
                print(f"   ✅ {label} 완료")
                return r
            except Exception as e:
                print(f"   ⚠️ {label} 실패: {e}")
                return default

        natam_result  = _get(fut_natam,  {"A":{},"B":{},"overall_a":"SAFE","overall_b":"SAFE"}, "NATAM")
        search_result = _get(fut_search, ([], [], ""), "유사사례 검색")
        video_frames  = _get(fut_frames, [], "키프레임 분석")
        summary_data  = _get(
            fut_summary,
            {"summary_title":"영상 내용 분석 결과","incident_overview":"전사 데이터를 기반으로 분석되었습니다."},
            "사건 개요",
        )
        claim_result  = _get(fut_claims, {"claims": [], "count": 0}, "검증대상 주장(Stage1)")
        bg_exec.shutdown(wait=False)

        cases, dists, summary = search_result if isinstance(search_result, tuple) else ([], [], "")
        print(f"   NATAM — A축: {natam_result['overall_a']} | B축: {natam_result['overall_b']}")

        # ── [하위] 룰스캔 union: raw + refined + 화면 OCR 텍스트 ──
        ocr_screen_text = " ".join(
            f.get("screen_text", "") for f in (video_frames or []) if f.get("screen_text")
        )
        rule_result = self.rule_scan_union(raw_transcript_text, transcript_text, ocr_screen_text)
        if rule_result.get("hit"):
            print(f"   🚨 룰 매칭 {len(rule_result.get('all_matches', []))}건 "
                  f"(raw+refined+OCR union)")

        # ── [2순위] 2-Tier 분류: SVM 트리거 → LLM 맥락 판정 ──
        similar_titles = [c.get("title", "") for c in cases]
        rule_words     = [m["matched_word"] for m in rule_result.get("all_matches", [])]
        two_tier_result = self.two_tier_classify(
            refined_segments, rule_hits=rule_words, similar_titles=similar_titles,
        )

        # ── 확산 단계 판정 ────────────────────────────────
        if youtube_meta:
            spread_result = analyze_spread_from_youtube_meta(youtube_meta)
            print(f"   📊 확산 단계 (유튜브 실측): {spread_result['stage']}")
        else:
            spread_result = {"stage": "Early", "reasons": [], "metrics": None}
            if cases and cases[0].get("incident_metadata"):
                spread_result = self.spread_analyzer.predict_stage(cases[0])

        detected_type = cases[0].get("controversy_type", "") if cases else ""
        worst_actions = self.get_risk_guide(detected_type, spread_result["stage"].lower())

        elapsed = round(time.time() - t0, 1)
        print(f"\n⏱️  총 분석 시간: {elapsed}초")

        # ── 보고서 구성 ────────────────────────────────────
        report = {
            "meta": {
                "video_filename":   os.path.basename(video_path),
                "analyzed_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "whisper_model":    WHISPER_MODEL,
                "duration_sec":     raw["video_info"]["duration"],
                "total_segments":   len(refined_segments),
                "total_corrected":  refined_data["total_corrected"],
                "input_query":      query,
                "incident_title":   summary_data["summary_title"],
                "incident_summary": summary_data["incident_overview"],
                "youtube_url":      youtube_meta.get("url","") if youtube_meta else "",
                "youtube_channel":  youtube_meta.get("channel","") if youtube_meta else "",
                "elapsed_sec":      elapsed,
                "transcribe_engine": TRANSCRIBE_ENGINE,
                "cost":             self.cost.summary(),
            },
            "rule_scan":      rule_result,
            "two_tier":       two_tier_result,
            "spread_stage":   spread_result,
            "classification": classification,
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
            "pattern_summary":     summary.strip() if isinstance(summary, str) else "",
            "worst_actions":       worst_actions,
            "transcript_analysis": transcript_analysis,
            "keyframe_analysis":   video_frames,
            "transcript_files": {
                "raw":     raw_path,
                "refined": refined_path,
            },
            "youtube_meta": youtube_meta or {},
            "natam_risk":   natam_result,
            "claim_check":  claim_result,   # [L04 Stage1] 검증 필요 주장(진위 판정 없음)
        }

        # ── 저장 ───────────────────────────────────────────
        os.makedirs(self.report_dir, exist_ok=True)
        json_path = os.path.join(self.report_dir, f"report_{base}.json")
        _save_json(report, json_path)
        print(f"\n💾 JSON 저장 완료 → {json_path}")

        print("📄 MD 리포트 생성 중...")
        md_path = self.report_engine.create_report(report)

        pdf_path = None
        if _PDF_AVAILABLE:
            try:
                print("📑 PDF 리포트 생성 중...")
                pdf_path = _gen_pdf(report, output_dir=self.report_dir)
            except Exception as e:
                print(f"⚠️ PDF 생성 실패: {e}")
        else:
            print("⚠️ pdf_report_generator.py 가 없어 PDF 생성을 건너뜁니다.")

        return report, json_path, md_path, pdf_path


# ════════════════════════════════════════════════════════
# 🖥️  터미널 인터페이스
# ════════════════════════════════════════════════════════

def print_report(report: dict):
    meta = report["meta"]
    yt   = report.get("youtube_meta", {})

    print(f"\n{'='*25} 📋 분석 요약 {'='*25}")
    if "video_filename" in meta:
        print(f"🎬 영상  : {meta['video_filename']}")
        if yt.get('url'):
            print(f"🔗 URL   : {yt['url']}")
            print(f"📺 채널  : {yt.get('channel','')} (구독자 {yt.get('subscriber_count',0):,}명)")
            print(f"📊 지표  : 조회수 {yt.get('view_count',0):,} | 좋아요 {yt.get('like_count',0):,} | 댓글 {yt.get('comment_count',0):,}")
        print(f"🕐 시각  : {meta['analyzed_at']}")
        print(f"⏱️  길이  : {meta.get('duration_sec','—')}초 | 세그먼트: {meta.get('total_segments','—')}개 | 교정: {meta.get('total_corrected','—')}개")
        print(f"⚡ 분석 소요: {meta.get('elapsed_sec','—')}초")
    else:
        print(f"🎯 입력  : \"{meta.get('input_query','')}\"")
        print(f"🕐 시각  : {meta['analyzed_at']}")
        if meta.get('elapsed_sec') is not None:
            print(f"⚡ 분석 소요: {meta.get('elapsed_sec','—')}초")

    r = report["rule_scan"]
    if r.get("hit"):
        print(f"\n🚨 [룰 적발] {r['policy']} ({r['severity']}) → {r['action']}  원인어: '{r['matched_word']}'")

    print("\n📂 [유사 사례 Top 3]")
    for c in report["similar_cases"]:
        print(f"   {c['rank']}. [{c['controversy_type']}] {c['title']}  (거리: {c['distance']})")
        print(f"      💡 대응: {c.get('response_pattern', [])}")

    cl = report["classification"]
    print(f"\n🏷️  [논란 유형] 주요: {cl.get('primary','')}  |  라벨: {', '.join(cl.get('labels',[]))}")
    print(f"   근거: {cl.get('reason','')}")

    sp = report["spread_stage"]
    source_tag = " (유튜브 실측)" if sp.get("source") == "youtube_meta" else ""
    print(f"\n📈 [확산 단계: {sp['stage']}{source_tag}]")
    for reason in sp["reasons"]: print(f"   • {reason}")

    print("\n📊 [공통 패턴 요약]")
    print(report["pattern_summary"])

    if report.get("worst_actions"):
        print("\n⛔ [금기 행동 가이드]")
        for i, act in enumerate(report["worst_actions"], 1):
            print(f"   {i}. {act}")

    ta = report.get("transcript_analysis", [])
    # L04(허위 정보)는 '검증 필요 주장' 섹션에서 따로 다루므로 주요 발언에선 제외
    ta_disp = [s for s in ta if not s.get("label", "").startswith("L04")]
    if ta_disp:
        print("\n📝 [자막 분석 (상위 5건)]")
        for item in ta_disp[:5]:
            print(f"   [{item.get('label','?')}] {item.get('corrected_text','')}")
            print(f"          → {item.get('reason','')}")

    cc = report.get("claim_check", {})
    claims = cc.get("claims", []) if isinstance(cc, dict) else []
    if claims:
        _cw = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        print(f"\n🔎 [검증 필요 주장 {len(claims)}건 — Stage1, 진위 판정 아님]")
        for c in claims[:5]:
            mark = _cw.get(c.get("checkworthiness"), "⚪")
            print(f"   {mark} [{c.get('timestamp','—')}] ({c.get('domain','기타')}) {c.get('claim','')}")
            if c.get("reason"):
                print(f"          → {c['reason']}")
        if len(claims) > 5:
            print(f"   … 외 {len(claims) - 5}건 (전체는 리포트 참고)")
        print("   ⚠️ 위 항목은 '검증이 필요한 주장'이며, 거짓으로 단정한 것이 아닙니다.")

    kf = report.get("keyframe_analysis", [])
    if kf:
        print("\n🎞️  [키프레임 분석]")
        for f in kf: print(f"   📍 {f['time']}: {f['tag']}")

    tf = report.get("transcript_files", {})
    if any(tf.values()):
        print("\n📁 [생성된 자막 파일]")
        for k, v in tf.items():
            if v: print(f"   {k}: {v}")

    # NATAM v2.0 리스크 요약
    natam = report.get("natam_risk", {})
    if natam:
        oa = natam.get("overall_a", "—")
        ob = natam.get("overall_b", "—")
        print(f"\n🛡️  [NATAM v2.0 리스크 요약]")
        print(f"   A축 커뮤니티 종합: {NATAM_LEVEL_EMOJI.get(oa,'⚪')} {oa}")
        for k, v in natam.get("A", {}).items():
            lvl   = v.get("level", "SAFE")
            emoji = NATAM_LEVEL_EMOJI.get(lvl, "⚪")
            name  = NATAM_A_AXES.get(k, {}).get("name", k)
            print(f"     {k} {name:<20} {emoji} {lvl}")
        print(f"   B축 플랫폼  종합: {NATAM_LEVEL_EMOJI.get(ob,'⚪')} {ob}")
        for k, v in natam.get("B", {}).items():
            lvl   = v.get("level", "SAFE")
            emoji = NATAM_LEVEL_EMOJI.get(lvl, "⚪")
            name  = NATAM_B_AXES.get(k, {}).get("name", k)
            print(f"     {k} {name:<20} {emoji} {lvl}")

    print("=" * 63)


def main():
    print("=" * 63)
    print("🛡️  CRISIS CONSULTANT v1.9 — 크리에이터 위기 관리 시스템")
    print("     ML 분류 엔진 통합 | NATAM v2.0 | Gemini 전사 | 유튜브 URL · 로컬 파일 지원")
    print("=" * 63)

    system = CrisisConsultantSystem(
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
        print("유튜브 URL 입력      →  다운로드 + 실측 지표 + 전체 자동 분석")
        print("영상 파일 경로 입력  →  전사부터 MD 리포트까지 전체 자동 분석")
        print("텍스트 입력          →  사건 설명 직접 분석 + MD 리포트 생성")
        print("[q]                  →  종료")
        user_input = input("💬 입력: ").strip().replace('"', '')

        if user_input.lower() == 'q':
            print("👋 종료합니다.")
            break
        if not user_input:
            continue

        # ── 유튜브 URL ──────────────────────────────────────
        if is_youtube_url(user_input):
            report, json_path, md_path, pdf_path = system.analyze_video_full(
                video_input  = user_input,
                output_dir   = "samples/transcripts",
                download_dir = "downloads",
                use_cache    = True,
            )
            print_report(report)
            print(f"\n✅ JSON 보고서 : {json_path}")
            if md_path:  print(f"✅ MD  리포트  : {md_path}")
            if pdf_path: print(f"✅ PDF 리포트  : {pdf_path}")

        # ── 로컬 영상 파일 ──────────────────────────────────
        elif os.path.splitext(user_input)[1].lower() in VIDEO_EXTS:
            if not os.path.exists(user_input):
                print("❌ 파일을 찾을 수 없습니다.")
                continue
            report, json_path, md_path, pdf_path = system.analyze_video_full(
                video_input  = user_input,
                output_dir   = "samples/transcripts",
                use_cache    = True,
            )
            print_report(report)
            print(f"\n✅ JSON 보고서 : {json_path}")
            if md_path:  print(f"✅ MD  리포트  : {md_path}")
            if pdf_path: print(f"✅ PDF 리포트  : {pdf_path}")

        # ── 텍스트 직접 입력 ────────────────────────────────
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
            natam_result = assess_natam_risk(
                client             = system.client,
                system_instruction = system.system_instruction,
                transcript_text    = "",
                summary            = query,
                gen_model          = GEN_MODEL,
            )
            print(f"   A축 종합: {natam_result['overall_a']} | B축 종합: {natam_result['overall_b']}")

            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            report = {
                "meta": {
                    "input_query": query,
                    "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            }

            os.makedirs("reports", exist_ok=True)
            json_path = os.path.join("reports", f"report_{ts}.json")
            _save_json(report, json_path)

            print("📄 MD 리포트 생성 중...")
            md_path = system.report_engine.create_report(report)

            print_report(report)
            print(f"\n✅ JSON 보고서 : {json_path}")
            if md_path: print(f"✅ MD  리포트  : {md_path}")


if __name__ == "__main__":
    main()