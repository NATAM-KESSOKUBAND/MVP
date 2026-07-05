"""
pipeline.py - 메인 분석 파이프라인 오케스트레이터
목표: AWS 환경에서 8~12분 내 완료
병렬 처리로 최대 효율
"""
import asyncio
import time
import uuid
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import structlog

from config import config
from database.db_manager import get_db_manager
from database.models import AnalysisStatus, RiskLevel
from youtube_risk import summarize_youtube_impact, youtube_impact_for_finding
from utils.video_utils import (
    compute_video_hash, get_video_info,
    extract_audio, extract_frames_smart, cleanup_temp_files,
)
from utils.ocr_engine import clear_ocr_cache
from analyzers.music_analyzer import MusicAnalyzer
from analyzers.image_analyzer import ImageAnalyzer
from analyzers.font_analyzer import FontAnalyzer
from analyzers.video_clip_analyzer import VideoClipAnalyzer

logger = structlog.get_logger()


# ─────────────────────────────────────────────
# 결과 집계
# ─────────────────────────────────────────────
def aggregate_risk(findings: List[Dict]) -> Tuple[float, RiskLevel]:
    """
    전체 저작권 위험도 집계
    분류별 가중치 적용
    """
    if not findings:
        return 0.0, RiskLevel.SAFE

    type_scores = {}
    for finding in findings:
        ftype = finding.get("finding_type", "image")
        risk = finding.get("risk_score", 0.0)
        if ftype not in type_scores:
            type_scores[ftype] = []
        type_scores[ftype].append(risk)

    # 각 타입의 최고 위험도 평균 (최악의 경우 반영)
    weights = config.risk.WEIGHTS
    total_weight = 0
    weighted_sum = 0

    for ftype, scores in type_scores.items():
        max_score = max(scores)
        weight = weights.get(ftype, 0.1)
        weighted_sum += max_score * weight
        total_weight += weight

    # 정규화
    if total_weight > 0:
        overall = weighted_sum / total_weight
        # 발견 수에 따른 보정
        count_bonus = min(len(findings) * 0.02, 0.15)
        overall = min(overall + count_bonus, 1.0)
    else:
        overall = 0.0

    if overall >= config.risk.HIGH_THRESHOLD:
        level = RiskLevel.HIGH
    elif overall >= config.risk.MEDIUM_THRESHOLD:
        level = RiskLevel.MEDIUM
    elif overall >= config.risk.LOW_THRESHOLD:
        level = RiskLevel.LOW
    else:
        level = RiskLevel.SAFE

    return overall, level


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _merge_cluster(cluster: List[Dict]) -> Dict:
    """시간상 인접한 동일 출처 finding 묶음 → 하나의 구간 finding"""
    if len(cluster) == 1:
        return cluster[0]
    best = max(cluster, key=lambda f: f.get("risk_score", 0))
    start = min(f.get("timestamp_start", 0) for f in cluster)
    end = max(f.get("timestamp_end") or f.get("timestamp_start", 0) for f in cluster)
    merged = dict(best)
    merged["timestamp_start"] = start
    merged["timestamp_end"] = end
    merged["timestamp_display"] = f"{_fmt_ts(start)}~{_fmt_ts(end)}"
    merged["confidence_score"] = round(
        min(max(f.get("confidence_score", 0) for f in cluster)
            * (1 + 0.05 * (len(cluster) - 1)), 0.97), 3)
    merged["description"] = (
        best.get("description", "")
        + f" [{_fmt_ts(start)}~{_fmt_ts(end)} 구간에서 {len(cluster)}회 감지]"
    )
    return merged


def merge_adjacent_findings(findings: List[Dict], max_gap: float = 90.0) -> List[Dict]:
    """
    같은 출처의 시각 finding이 시간상 인접하면 하나의 구간으로 병합.

    점(point) 단위 finding이 흩어져 리포트가 산만해지는 것을 막고
    '구간 + 반복 횟수'라는 더 강한 증거 형태로 제시한다.
    music은 music_analyzer가 자체적으로 구간 병합하므로 제외.
    """
    MERGE_TYPES = {"video_clip", "image", "logo"}

    def _key(f: Dict):
        holder = f.get("rights_holder") or ""
        if holder:
            base = holder
        else:
            title = f.get("title", "")
            base = (title.split(": ", 1)[-1].split(",")[0].strip()
                    if ": " in title else title)
        return (f.get("finding_type"), f.get("source", ""), base)

    mergeable = sorted(
        (f for f in findings if f.get("finding_type") in MERGE_TYPES),
        key=lambda x: x.get("timestamp_start", 0),
    )
    out = [f for f in findings if f.get("finding_type") not in MERGE_TYPES]

    groups: Dict = {}
    for f in mergeable:
        groups.setdefault(_key(f), []).append(f)

    for group in groups.values():
        cluster = [group[0]]
        for f in group[1:]:
            prev_end = cluster[-1].get("timestamp_end") \
                       or cluster[-1].get("timestamp_start", 0)
            if f.get("timestamp_start", 0) - prev_end <= max_gap:
                cluster.append(f)
            else:
                out.append(_merge_cluster(cluster))
                cluster = [f]
        out.append(_merge_cluster(cluster))

    out.sort(key=lambda x: x.get("timestamp_start", 0))
    return out


def format_results(job_id: str, video_path: str, video_info: Dict,
                   all_findings: List[Dict], processing_time: float) -> Dict:
    """최종 결과 딕셔너리 구성"""
    risk_score, risk_level = aggregate_risk(all_findings)

    # 타입별 그룹핑
    by_type = {}
    for f in all_findings:
        ftype = f.get("finding_type", "unknown")
        if ftype not in by_type:
            by_type[ftype] = []
        by_type[ftype].append(f)

    # 타임라인 (시간순 정렬)
    timeline = sorted(all_findings, key=lambda x: x.get("timestamp_start", 0))

    return {
        "job_id": job_id,
        "video_path": video_path,
        "video_filename": os.path.basename(video_path),
        "video_duration": video_info.get("duration", 0),
        "video_info": video_info,
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "processing_time_sec": processing_time,

        "summary": {
            "overall_risk_score": round(risk_score * 100, 1),   # % 형태
            "overall_risk_level": risk_level.value,
            "total_issues_found": len(all_findings),
            "by_type": {
                ftype: len(findings)
                for ftype, findings in by_type.items()
            },
            "high_risk_count": sum(1 for f in all_findings if f.get("risk_level") == "HIGH"),
            "medium_risk_count": sum(1 for f in all_findings if f.get("risk_level") == "MEDIUM"),
            # 유튜브 스튜디오 관점 예측 (노란 딱지/클레임/차단/경고 가능성)
            "youtube": summarize_youtube_impact(all_findings),
        },

        "findings_by_type": {
            "music": _format_findings(by_type.get("music", [])),
            "video_clip": _format_findings(by_type.get("video_clip", [])),
            "image": _format_findings(by_type.get("image", [])),
            "logo": _format_findings(by_type.get("logo", [])),
            "font": _format_findings(by_type.get("font", [])),
        },

        "timeline": _format_findings(timeline),
    }


def _format_findings(findings: List[Dict]) -> List[Dict]:
    """결과 데이터 정리 (raw_response 제거 등) + 유튜브 조치 예측 부착"""
    clean = []
    for f in findings:
        yt = youtube_impact_for_finding(f)
        clean.append({
            "timestamp": f.get("timestamp_display", "00:00"),
            "timestamp_start_sec": f.get("timestamp_start", 0),
            "timestamp_end_sec": f.get("timestamp_end", 0),
            "type": f.get("finding_type", ""),
            "title": f.get("title", ""),
            "author": f.get("author", ""),
            "rights_holder": f.get("rights_holder", ""),
            "source": f.get("source", ""),
            "confidence": f"{round(f.get('confidence_score', 0) * 100, 1)}%",
            "risk_score": f"{round(f.get('risk_score', 0) * 100, 1)}%",
            "risk_level": f.get("risk_level", "SAFE"),
            # 유튜브 스튜디오 관점: 이 항목이 실제로 어떤 조치를 부를지
            "yt_outcome": yt["yt_outcome"],
            "yt_outcome_label": yt["yt_outcome_label"],
            "yt_emoji": yt["yt_emoji"],
            "yt_claim_prob": f"{round(yt['yt_claim_prob'] * 100)}%",
            "description": f.get("description", ""),
            "reference_url": f.get("reference_url"),
        })
    return clean


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────
class CopyrightAnalysisPipeline:
    """
    저작권 분석 파이프라인

    실행 순서 (병렬):
    1. 영상 정보 + 해시 계산 (직렬, 빠름)
    2. 캐시 확인 (직렬, 빠름)
    3. 오디오 추출 + 프레임 추출 (병렬)
    4. 분석 5개 모듈 (병렬) ← 대부분 시간
    5. 결과 집계 + DB 저장 (직렬, 빠름)
    6. 리포트 생성 (직렬)

    타이밍 목표 (AWS ECS Fargate 4vCPU):
    - 영상 길이 ~30분: 8~10분
    - 영상 길이 ~60분: 10~12분
    """

    def __init__(self):
        self.db = get_db_manager()
        self.music_analyzer = MusicAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.font_analyzer = FontAnalyzer()
        self.video_clip_analyzer = VideoClipAnalyzer()

    async def run(self, video_path: str,
                  job_id: Optional[str] = None,
                  force_reanalyze: bool = False) -> Dict:
        """
        메인 실행 함수

        Args:
            video_path: 영상 파일 경로 (로컬 또는 S3 마운트 경로)
            job_id: 작업 ID (없으면 자동 생성)
            force_reanalyze: True면 캐시 무시

        Returns:
            분석 결과 딕셔너리
        """
        start_time = time.time()
        job_id = job_id or str(uuid.uuid4())[:8].upper()

        # 작업 시작 시 OCR 캐시 초기화 (이전 작업 캐시 누적 방지)
        clear_ocr_cache()

        # 본인 소유 콘텐츠 필터 적용 (.env의 OWN_CHANNELS / OWN_DOMAINS)
        # → Vision이 본인 유튜브/사이트의 자기 콘텐츠를 위반으로 오탐하는 것 방지
        try:
            from utils.google_vision_searcher import set_own_channels, set_own_domains
            if config.own.channels:
                set_own_channels(config.own.channels)
            if config.own.domains:
                set_own_domains(config.own.domains)
        except Exception as e:
            logger.warning("own_content_filter_setup_failed", error=str(e))

        logger.info("pipeline_start", job_id=job_id, video=video_path)

        # ─── Step 1: 영상 기본 정보 ───
        logger.info("step1_video_info", job_id=job_id)
        t1 = time.time()
        video_hash = compute_video_hash(video_path)
        video_info = get_video_info(video_path)
        duration = video_info.get("duration", 0)
        logger.info("video_info_done",
                    duration=f"{duration:.0f}s",
                    hash=video_hash[:8],
                    elapsed=f"{time.time()-t1:.1f}s")

        # ─── Step 2: 캐시 확인 ───
        if not force_reanalyze:
            cached = self.db.check_video_cached(video_hash)
            if cached:
                logger.info("cache_hit_returning", job_id=job_id)
                return cached

        # ─── Step 3: DB 작업 생성 ───
        self.db.create_job(
            job_id=job_id,
            video_path=video_path,
            video_hash=video_hash,
            video_duration=duration,
            metadata=video_info,
        )

        try:
            # ─── Step 4: 오디오 + 프레임 병렬 추출 ───
            logger.info("step4_extraction", job_id=job_id)
            t4 = time.time()

            audio_path, frames = await asyncio.gather(
                asyncio.to_thread(
                    extract_audio,
                    video_path,
                    str(config.TEMP_DIR / f"{job_id}_audio.wav")
                ),
                asyncio.to_thread(
                    extract_frames_smart,
                    video_path,
                    config.pipeline.frame_extraction_fps,
                    config.pipeline.frame_phash_threshold,
                    config.pipeline.frame_max_count,
                )
            )

            logger.info("extraction_done",
                        audio=audio_path,
                        frames=len(frames),
                        elapsed=f"{time.time()-t4:.1f}s")

            # ─── Step 5: 5개 분석 모듈 완전 병렬 ───
            logger.info("step5_parallel_analysis", job_id=job_id, modules=5)
            t5 = time.time()

            # audio_path 전달 → 음악 분석이 추출된 wav를 메모리 슬라이스
            # (청크당 ffmpeg 프로세스 기동 제거: 30분 영상 기준 ~180회 → 0회)
            music_task      = self.music_analyzer.analyze(video_path, job_id,
                                                          audio_path=audio_path)
            image_task      = self.image_analyzer.analyze(frames, job_id)
            font_task       = self.font_analyzer.analyze(frames, job_id)
            video_clip_task = self.video_clip_analyzer.analyze(frames, job_id)

            # 모든 분석 동시 실행
            music_findings, image_findings, font_findings, video_findings = \
                await asyncio.gather(
                    music_task,
                    image_task,
                    font_task,
                    video_clip_task,
                    return_exceptions=True
                )

            # 에러 처리
            all_findings = []
            for name, result in [
                ("music", music_findings),
                ("image", image_findings),
                ("font", font_findings),
                ("video_clip", video_findings),
            ]:
                if isinstance(result, Exception):
                    logger.error(f"{name}_analyzer_failed", error=str(result))
                elif isinstance(result, list):
                    all_findings.extend(result)

            logger.info("parallel_analysis_done",
                        total_findings=len(all_findings),
                        elapsed=f"{time.time()-t5:.1f}s")

            # ─── Step 5.5: 인접 동일 출처 finding 구간 병합 ───
            n_before = len(all_findings)
            all_findings = merge_adjacent_findings(all_findings)
            if len(all_findings) < n_before:
                logger.info("findings_merged",
                            before=n_before, after=len(all_findings))

            # ─── Step 6: 결과 집계 ───
            processing_time = time.time() - start_time
            results = format_results(
                job_id, video_path, video_info, all_findings, processing_time
            )

            # ─── Step 7: DB 저장 ───
            logger.info("step7_db_save", job_id=job_id)
            risk_score = results["summary"]["overall_risk_score"] / 100.0
            risk_level_str = results["summary"]["overall_risk_level"]

            # Findings 저장
            if all_findings:
                self.db.save_findings(all_findings)

            # Job 완료 업데이트
            self.db.update_job_status(
                job_id=job_id,
                status=AnalysisStatus.COMPLETED,
                risk_score=risk_score,
                risk_level=RiskLevel(risk_level_str),
                total_issues=len(all_findings),
            )

            # 리포트 저장
            self.db.save_report(job_id, results)

            total_time = time.time() - start_time
            logger.info("pipeline_complete",
                        job_id=job_id,
                        total_sec=f"{total_time:.1f}s",
                        total_min=f"{total_time/60:.1f}min",
                        findings=len(all_findings),
                        risk=risk_level_str)

            return results

        except Exception as e:
            logger.error("pipeline_failed", job_id=job_id, error=str(e))
            self.db.update_job_status(
                job_id=job_id,
                status=AnalysisStatus.FAILED,
                error=str(e),
            )
            raise

        finally:
            # 임시 파일 정리
            cleanup_temp_files(prefix=f"{job_id}_")
            cleanup_temp_files(prefix="chunk_")


# ─────────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────────
async def analyze_video(video_path: str, job_id: str = None,
                        force_reanalyze: bool = False) -> Dict:
    """단일 영상 분석 (외부 호출용)"""
    pipeline = CopyrightAnalysisPipeline()
    return await pipeline.run(video_path, job_id, force_reanalyze=force_reanalyze)


def analyze_video_sync(video_path: str, job_id: str = None,
                       force_reanalyze: bool = False) -> Dict:
    """동기 래퍼 (AWS Lambda, CLI 등에서 사용)"""
    return asyncio.run(analyze_video(video_path, job_id, force_reanalyze=force_reanalyze))
