"""
analyzers/music_analyzer.py - 음악 저작권 분석
우선순위: 자체 DB → ACRCloud → AudD (백업)
"""
import os
import asyncio
import hashlib
import base64
import time
import hmac
import struct
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import structlog
import requests

from config import config
from database.db_manager import get_db_manager
from utils.video_utils import get_audio_chunks, get_video_info

logger = structlog.get_logger()


# ─────────────────────────────────────────────
# 위험도 계산
# ─────────────────────────────────────────────
def calculate_music_risk(confidence: float, source: str,
                          detection_count: int = 1) -> Tuple[float, str]:
    """
    음악 저작권 위험도 계산
    - confidence: API 신뢰도 (0~1)
    - source: 데이터 소스
    - detection_count: 자체 DB에서 몇 번 발견됐는지
    """
    base_risk = confidence

    # 자체 DB에서 여러 번 발견된 경우 위험도 상승
    if detection_count > 5:
        base_risk = min(base_risk * 1.15, 1.0)
    elif detection_count > 1:
        base_risk = min(base_risk * 1.05, 1.0)

    # 소스별 신뢰도 가중치
    source_weights = {
        "internal_db": 1.10,  # 자체 DB는 이미 검증된 데이터
        "acrcloud": 1.00,
        "audd": 0.95,
        "manual": 1.10,
    }
    weight = source_weights.get(source, 1.0)
    final_risk = min(base_risk * weight, 1.0)

    # 등급 결정
    if final_risk >= config.risk.HIGH_THRESHOLD:
        level = "HIGH"
    elif final_risk >= config.risk.MEDIUM_THRESHOLD:
        level = "MEDIUM"
    elif final_risk >= config.risk.LOW_THRESHOLD:
        level = "LOW"
    else:
        level = "SAFE"

    return final_risk, level


# ─────────────────────────────────────────────
# ACRCloud 클라이언트
# ─────────────────────────────────────────────
class ACRCloudClient:
    """ACRCloud 음악 인식 API"""

    def __init__(self):
        self.host = config.api.acrcloud_host
        self.access_key = config.api.acrcloud_access_key
        self.access_secret = config.api.acrcloud_access_secret
        self.timeout = config.pipeline.api_timeout_seconds

    def _build_signature(self, timestamp: int) -> str:
        string_to_sign = f"POST\n/v1/identify\n{self.access_key}\naudio\n1\n{timestamp}"
        secret = self.access_secret.encode("utf-8")
        sig = hmac.new(secret, string_to_sign.encode("utf-8"), "sha1")
        return base64.b64encode(sig.digest()).decode("utf-8")

    def identify_from_file(self, audio_path: str) -> Optional[Dict]:
        """오디오 파일에서 음악 식별"""
        if not all([self.host, self.access_key, self.access_secret]):
            logger.warning("acrcloud_not_configured")
            return None

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            timestamp = int(time.time())
            signature = self._build_signature(timestamp)

            files = {"sample": audio_data}
            data = {
                "access_key": self.access_key,
                "sample_bytes": len(audio_data),
                "timestamp": timestamp,
                "signature": signature,
                "data_type": "audio",
                "signature_version": "1",
            }

            url = f"https://{self.host}/v1/identify"
            response = requests.post(url, files=files, data=data, timeout=self.timeout)
            result = response.json()

            if result.get("status", {}).get("code") == 0:
                return self._parse_result(result)
            return None

        except Exception as e:
            logger.error("acrcloud_error", error=str(e))
            return None

    def _parse_result(self, result: Dict) -> Dict:
        """ACRCloud 응답 파싱"""
        metadata = result.get("metadata", {})
        music_list = metadata.get("music", [])

        if not music_list:
            return None

        music = music_list[0]
        score = music.get("score", 0) / 100.0  # 0~100 → 0~1

        # ISRC 추출
        isrc = None
        for external in music.get("external_ids", {}).values():
            if isinstance(external, dict) and "isrc" in external:
                isrc = external["isrc"]
                break

        # 스트리밍 링크
        streaming = music.get("external_metadata", {})
        spotify_id = streaming.get("spotify", {}).get("track", {}).get("id")

        return {
            "title": music.get("title", ""),
            "artist": ", ".join(a.get("name", "") for a in music.get("artists", [])),
            "album": music.get("album", {}).get("name", ""),
            "rights_holder": music.get("label", ""),
            "isrc": isrc,
            "confidence": score,
            "source": "acrcloud",
            "acrcloud_id": music.get("acrid"),
            "spotify_id": spotify_id,
            "release_date": music.get("release_date", ""),
            "raw": result,  # 학습용 원본 저장
        }


# ─────────────────────────────────────────────
# AudD 클라이언트 (백업)
# ─────────────────────────────────────────────
class AudDClient:
    """AudD 음악 인식 API (ACRCloud 백업)"""

    def __init__(self):
        self.api_token = config.api.audd_api_token
        self.base_url = "https://api.audd.io/"

    def identify_from_file(self, audio_path: str) -> Optional[Dict]:
        if not self.api_token:
            return None

        try:
            with open(audio_path, "rb") as f:
                files = {"file": f}
                data = {
                    "api_token": self.api_token,
                    "return": "spotify,apple_music",
                }
                response = requests.post(
                    self.base_url, files=files, data=data,
                    timeout=config.pipeline.api_timeout_seconds
                )

            result = response.json()
            if result.get("status") == "success" and result.get("result"):
                return self._parse_result(result["result"])
            return None

        except Exception as e:
            logger.error("audd_error", error=str(e))
            return None

    def _parse_result(self, result: Dict) -> Dict:
        return {
            "title": result.get("title", ""),
            "artist": result.get("artist", ""),
            "album": result.get("album", ""),
            "rights_holder": result.get("label", ""),
            "isrc": result.get("spotify", {}).get("external_ids", {}).get("isrc"),
            "confidence": 0.80,  # AudD는 confidence 미제공 → 기본값
            "source": "audd",
            "raw": result,
        }


# ─────────────────────────────────────────────
# 메인 음악 분석기
# ─────────────────────────────────────────────
class MusicAnalyzer:
    """
    병렬 음악 저작권 분석
    파이프라인: 자체DB → ACRCloud → AudD
    """

    def __init__(self):
        self.acrcloud = ACRCloudClient()
        self.audd = AudDClient()
        self.db = get_db_manager()
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def analyze(self, video_path: str, job_id: str) -> List[Dict]:
        """
        영상의 전체 오디오를 청크 단위로 병렬 분석
        Returns: 발견된 저작권 음악 목록
        """
        logger.info("music_analysis_start", job_id=job_id)

        # 오디오 청크 생성 + 병렬 API 호출
        loop = asyncio.get_event_loop()
        tasks = []

        chunks = list(get_audio_chunks(
            video_path,
            chunk_duration=config.pipeline.audio_fingerprint_duration,
            overlap=config.pipeline.audio_chunk_overlap
        ))

        logger.info("audio_chunks_created", count=len(chunks), job_id=job_id)

        # 병렬 처리 (최대 4개 동시)
        semaphore = asyncio.Semaphore(4)

        async def analyze_chunk(start_time: float, chunk_path: str):
            async with semaphore:
                return await loop.run_in_executor(
                    self.executor,
                    self._analyze_single_chunk,
                    start_time, chunk_path, job_id
                )

        results = await asyncio.gather(
            *[analyze_chunk(start, path) for start, path in chunks],
            return_exceptions=True
        )

        hits = [r for r in results if r and not isinstance(r, Exception)]
        findings = self._build_findings_from_hits(hits, job_id)

        logger.info("music_analysis_done", findings=len(findings), job_id=job_id)
        return findings

    def _build_findings_from_hits(self, hits: List[Dict], job_id: str) -> List[Dict]:
        """
        청크 단위 매칭 결과 → 곡별 연속 구간(run) 병합 → finding 생성.

        기존 방식은 곡당 첫 번째 청크만 남겨(ISRC dedup) 사용 구간 정보가
        사라졌다. 이제 인접 청크를 구간으로 병합하고 '얼마나 길게 썼는지'를
        위험도에 반영한다 (3초 BGM ≠ 3분 BGM).
        """
        findings: List[Dict] = []
        if not hits:
            return findings

        chunk_dur = config.pipeline.audio_fingerprint_duration
        step = max(chunk_dur - config.pipeline.audio_chunk_overlap, 1)

        def _song_key(r: Dict) -> str:
            return r.get("isrc") or f"{r.get('title', '')}_{r.get('artist', '')}"

        by_song: Dict[str, List[Dict]] = {}
        for r in hits:
            by_song.setdefault(_song_key(r), []).append(r)

        for key, group in by_song.items():
            group.sort(key=lambda r: r["start_time"])

            # 인접 청크 병합: 청크 1개가 빠져도 (잡음 등) 같은 구간으로 간주
            runs: List[List[Dict]] = [[group[0]]]
            for r in group[1:]:
                if r["start_time"] - runs[-1][-1]["start_time"] <= step * 2 + 1:
                    runs[-1].append(r)
                else:
                    runs.append([r])

            for run in runs:
                best = max(run, key=lambda r: r.get("confidence", 0))
                isrc = best.get("isrc")
                span_start = run[0]["start_time"]
                span_end = run[-1]["start_time"] + chunk_dur
                duration = span_end - span_start

                risk_score, risk_level = calculate_music_risk(
                    best["confidence"],
                    best["source"],
                    max(best.get("detection_count", 1), len(run)),
                )

                # ── 구간 길이 가중: 길게 사용된 음악일수록 침해 심각도 ↑ ──
                if duration >= 60:
                    risk_score = min(risk_score * 1.15, 1.0)
                elif duration >= 30:
                    risk_score = min(risk_score * 1.08, 1.0)
                # 가중 후 등급 재산정
                if risk_score >= config.risk.HIGH_THRESHOLD:
                    risk_level = "HIGH"
                elif risk_score >= config.risk.MEDIUM_THRESHOLD:
                    risk_level = "MEDIUM"
                elif risk_score >= config.risk.LOW_THRESHOLD:
                    risk_level = "LOW"
                else:
                    risk_level = "SAFE"

                if risk_score < 0.15:  # 너무 낮으면 제외
                    continue

                desc = self._build_description(best)
                if len(run) >= 2:
                    desc += (
                        f" [{self._format_time(span_start)}~{self._format_time(span_end)}"
                        f" 약 {duration:.0f}초 사용, 청크 {len(run)}개 일치]"
                    )

                findings.append({
                    "job_id": job_id,
                    "finding_type": "music",
                    "timestamp_start": span_start,
                    "timestamp_end": span_end,
                    "timestamp_display": self._format_time(span_start),
                    "title": best.get("title", "Unknown"),
                    "author": best.get("artist", ""),
                    "rights_holder": best.get("rights_holder", ""),
                    "source": best["source"],
                    "external_id": isrc or best.get("acrcloud_id"),
                    "confidence_score": best["confidence"],
                    "risk_score": round(risk_score, 3),
                    "risk_level": risk_level,
                    "description": desc,
                    "raw_response": best.get("raw"),
                })

                # 자체 DB 학습 (fingerprint_hash 포함 → 내부 DB 조회 활성화)
                self.db.learn_from_finding("music", {
                    "isrc": isrc,
                    "title": best.get("title", ""),
                    "artist": best.get("artist", ""),
                    "album": best.get("album", ""),
                    "rights_holder": best.get("rights_holder", ""),
                    "source": best["source"],
                    "acrcloud_id": best.get("acrcloud_id"),
                    "fingerprint_hash": best.get("fingerprint_hash"),
                })

        findings.sort(key=lambda f: f["timestamp_start"])
        return findings

    def _analyze_single_chunk(self, start_time: float, chunk_path: str,
                               job_id: str) -> Optional[Dict]:
        """단일 청크 분석 (동기)"""
        try:
            # 1. 자체 DB 먼저 확인 (빠름)
            # fingerprint hash 생성
            with open(chunk_path, "rb") as f:
                fp_hash = hashlib.md5(f.read(4096)).hexdigest()  # 간이 해시

            cached = self.db.lookup_music_by_fingerprint(fp_hash)
            if cached and cached["found"]:
                logger.debug("internal_db_hit", title=cached["title"])
                return {**cached, "start_time": start_time, "confidence": 0.95,
                        "fingerprint_hash": fp_hash}

            # 2. ACRCloud (주력)
            # fingerprint_hash 동봉 → 학습 시 저장돼 다음 분석에서 내부 DB가 작동
            # (기존엔 학습에 해시가 빠져 lookup_music_by_fingerprint가 항상 미스)
            result = self.acrcloud.identify_from_file(chunk_path)
            if result and result["confidence"] >= config.pipeline.music_confidence_threshold:
                return {**result, "start_time": start_time, "fingerprint_hash": fp_hash}

            # 3. AudD (백업)
            result = self.audd.identify_from_file(chunk_path)
            if result and result["confidence"] >= 0.5:
                return {**result, "start_time": start_time, "fingerprint_hash": fp_hash}

            return None

        except Exception as e:
            logger.error("chunk_analysis_error", start=start_time, error=str(e))
            return None
        finally:
            # 임시 파일 정리
            try:
                os.remove(chunk_path)
            except Exception:
                pass

    def _format_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _build_description(self, result: Dict) -> str:
        title = result.get("title", "Unknown")
        artist = result.get("artist", "")
        rights = result.get("rights_holder", "")
        parts = [f"'{title}'"]
        if artist:
            parts.append(f"by {artist}")
        if rights:
            parts.append(f"(© {rights})")
        return " ".join(parts)
