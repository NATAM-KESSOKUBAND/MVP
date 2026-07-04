"""
utils/downloader.py - URL 영상 다운로드 유틸
yt-dlp 기반, tui.py의 download_video_with_progress()에서 사용.
"""
import os
from pathlib import Path
from typing import Dict

import structlog

logger = structlog.get_logger()


def get_video_info_from_url(url: str) -> Dict:
    """
    URL에서 영상 메타데이터만 가져오기 (실제 다운로드 없음).

    Returns:
        {
            "title":    str,
            "uploader": str,
            "duration": int,   # 초 단위
            "ext":      str,   # 예상 확장자
        }
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        raise RuntimeError("yt-dlp가 설치되지 않았습니다. pip install yt-dlp")

    ydl_opts = {
        "quiet":           True,
        "no_warnings":     True,
        "skip_download":   True,
        "noplaylist":      True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "title":    info.get("title", ""),
        "uploader": info.get("uploader", info.get("channel", "")),
        "duration": int(info.get("duration") or 0),
        "ext":      info.get("ext", "mp4"),
    }


def download_video(url: str, output_dir: str = "temp/downloads") -> str:
    """
    URL 영상을 output_dir에 다운로드하고 저장된 파일 경로를 반환.

    Args:
        url:        다운로드할 영상 URL
        output_dir: 저장 디렉터리 경로

    Returns:
        다운로드된 파일의 절대 경로 (str)
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        raise RuntimeError("yt-dlp가 설치되지 않았습니다. pip install yt-dlp")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 저장 경로 템플릿: output_dir/%(id)s.%(ext)s
    outtmpl = str(Path(output_dir) / "%(id)s.%(ext)s")

    ydl_opts = {
        "quiet":       True,
        "no_warnings": True,
        "noplaylist":  True,
        "outtmpl":     outtmpl,
        # 최대 화질 mp4 우선, 없으면 최선의 단일 스트림
        "format":      "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
    }

    saved_path: str = ""

    def _progress_hook(d: dict):
        nonlocal saved_path
        if d.get("status") == "finished":
            saved_path = d.get("filename", "")

    ydl_opts["progress_hooks"] = [_progress_hook]

    logger.info("download_start", url=url[:80], output_dir=output_dir)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # progress_hook이 경로를 못 잡은 경우 최근 파일로 폴백
    if not saved_path or not Path(saved_path).exists():
        files = sorted(
            Path(output_dir).glob("*.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}
        for f in files:
            if f.suffix.lower() in video_exts:
                saved_path = str(f)
                break

    if not saved_path or not Path(saved_path).exists():
        raise FileNotFoundError(f"다운로드 후 파일을 찾을 수 없습니다: {output_dir}")

    logger.info("download_done", path=saved_path)
    return saved_path
