"""
config.py - 전체 설정 관리
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_BASE_DIR = Path(__file__).parent

# GOOGLE_APPLICATION_CREDENTIALS may be a relative path in .env (portable
# across machines); google-cloud client libraries read this env var directly
# and need an absolute path regardless of the process's working directory.
_cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _cred_path and not os.path.isabs(_cred_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_BASE_DIR / _cred_path)

# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────
@dataclass
class APIConfig:
    # ACRCloud (음악 인식) - https://www.acrcloud.com
    acrcloud_host: str = os.getenv("ACRCLOUD_HOST", "")
    acrcloud_access_key: str = os.getenv("ACRCLOUD_ACCESS_KEY", "")
    acrcloud_access_secret: str = os.getenv("ACRCLOUD_ACCESS_SECRET", "")

    # AudD (음악 인식 백업) - https://audd.io
    audd_api_token: str = os.getenv("AUDD_API_TOKEN", "")

    # Google Cloud Vision (이미지/로고 분석)
    google_credentials_path: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # YouTube Data API (영상 Content ID)
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")

    # AWS
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region: str = os.getenv("AWS_REGION", "ap-northeast-2")  # 서울 리전

    # SerpAPI (밈 역이미지 검색) - https://serpapi.com
    serpapi_key: str = os.getenv("SERPAPI_KEY", "")


@dataclass
class OwnContentConfig:
    """
    본인 소유 콘텐츠 등록 (오탐 방지).
    Vision이 본인 유튜브/사이트에서 자기 콘텐츠를 찾아 저작권 위반으로
    오탐하는 것을 막는다. .env에 쉼표로 구분해 입력.
      OWN_CHANNELS=MyChannel,@myhandle
      OWN_DOMAINS=myblog.com,myname.tistory.com
    """
    channels: list = field(default_factory=lambda: [
        c.strip() for c in os.getenv("OWN_CHANNELS", "").split(",") if c.strip()
    ])
    domains: list = field(default_factory=lambda: [
        d.strip() for d in os.getenv("OWN_DOMAINS", "").split(",") if d.strip()
    ])


# ─────────────────────────────────────────────
# Database Config
# ─────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    # 로컬/프로덕션 DB (PostgreSQL on RDS)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./copyright_db.sqlite"  # 로컬 개발용 SQLite
    )
    # AWS RDS (프로덕션)
    rds_host: str = os.getenv("RDS_HOST", "")
    rds_port: int = int(os.getenv("RDS_PORT", "5432"))
    rds_db: str = os.getenv("RDS_DB", "copyright_detector")
    rds_user: str = os.getenv("RDS_USER", "")
    rds_password: str = os.getenv("RDS_PASSWORD", "")

    @property
    def production_url(self) -> str:
        if self.rds_host:
            return f"postgresql://{self.rds_user}:{self.rds_password}@{self.rds_host}:{self.rds_port}/{self.rds_db}"
        return self.database_url


# ─────────────────────────────────────────────
# Pipeline Config
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    # 오디오 추출 설정
    audio_sample_rate: int = 16000
    audio_chunk_duration: int = 10        # seconds per chunk for music recognition
    audio_chunk_overlap: int = 2          # overlap between chunks
    audio_fingerprint_duration: int = 12  # ACRCloud fingerprint length

    # 영상 프레임 추출 (extract_frames_smart 파라미터)
    frame_extraction_fps: float = 1.0     # target FPS (길이에 따라 자동 축소)
    frame_phash_threshold: int = 12       # 유사 프레임 스킵 threshold (64비트 중)
    frame_scene_threshold: float = 30.0   # scene change detection threshold
    frame_max_count: int = 320            # 최대 추출 프레임 수 (30분 영상 전체 커버: ~5.6초 간격)

    # 병렬 처리
    max_workers: int = 8                  # 병렬 워커 수
    batch_size: int = 10                  # 배치 처리 크기

    # 타임아웃 (AWS Lambda 고려)
    total_timeout_minutes: int = 10       # 전체 10분 목표
    api_timeout_seconds: int = 30

    # 신뢰도 임계값
    music_confidence_threshold: float = 0.6
    image_confidence_threshold: float = 0.7
    logo_confidence_threshold: float = 0.75


# ─────────────────────────────────────────────
# AWS Config
# ─────────────────────────────────────────────
@dataclass
class AWSConfig:
    region: str = os.getenv("AWS_REGION", "ap-northeast-2")

    # S3
    s3_bucket: str = os.getenv("S3_BUCKET", "copyright-detector-videos")
    s3_results_bucket: str = os.getenv("S3_RESULTS_BUCKET", "copyright-detector-results")

    # ECS Fargate (메인 실행 환경)
    ecs_cluster: str = os.getenv("ECS_CLUSTER", "copyright-detector-cluster")
    ecs_task_definition: str = os.getenv("ECS_TASK_DEFINITION", "copyright-detector-task")

    # Lambda (경량 태스크용)
    lambda_function: str = os.getenv("LAMBDA_FUNCTION", "copyright-detector")

    # Rekognition (AWS 이미지 분석)
    use_rekognition: bool = os.getenv("USE_REKOGNITION", "true").lower() == "true"

    # SQS (비동기 처리)
    sqs_queue_url: str = os.getenv("SQS_QUEUE_URL", "")

    # ElastiCache Redis (캐시)
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))


# ─────────────────────────────────────────────
# 저작권 위험도 기준
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    # 위험도 등급 (0.0 ~ 1.0)
    HIGH_THRESHOLD: float = 0.75      # 🔴 높음
    MEDIUM_THRESHOLD: float = 0.45    # 🟡 중간
    LOW_THRESHOLD: float = 0.20       # 🟢 낮음
    # 이하 = SAFE

    # 분류별 기본 가중치 (합산 시 사용)
    WEIGHTS = {
        "music": 0.35,      # 음악이 가장 위험
        "video_clip": 0.28, # 영상 클립
        "image": 0.22,      # 이미지/사진
        "logo": 0.09,       # 로고/상표
        "font": 0.06,       # 폰트
    }


# ─────────────────────────────────────────────
# 통합 설정
# ─────────────────────────────────────────────
class Config:
    api = APIConfig()
    db = DatabaseConfig()
    pipeline = PipelineConfig()
    aws = AWSConfig()
    risk = RiskConfig()
    own = OwnContentConfig()

    BASE_DIR = Path(__file__).parent
    TEMP_DIR = BASE_DIR / "temp"
    REPORTS_DIR = BASE_DIR / "results"

    @classmethod
    def ensure_dirs(cls):
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
config.ensure_dirs()
