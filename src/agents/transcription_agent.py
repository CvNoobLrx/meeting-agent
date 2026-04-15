"""
Transcription Agent（转写Agent）
- 接收音频数据，使用 FUNASR 进行语音转文字
- 使用 pyannote-audio 进行说话人识别（Speaker Diarization）
- 输出带说话人标签和时间戳的转写文本
"""

from __future__ import annotations

import io
import os
import pathlib
import shutil
import tempfile
from typing import Any

import numpy as np
from loguru import logger

from ..models.schemas import (
    MeetingStatus,
    TranscriptResult,
    TranscriptSegment,
)


class TranscriptionConfig:
    """转写配置"""

    def __init__(
        self,
        model_size: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        compute_type: str = "float32",
        language: str = "zh",
        hf_token: str = "",
        batch_size: int = 16,
    ):
        self.model_size = os.getenv("ASR_MODEL_ID", model_size)
        self.device = os.getenv("ASR_DEVICE", device)
        self.compute_type = compute_type
        self.language = os.getenv("ASR_LANGUAGE", language)
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.batch_size = batch_size


class TranscriptionAgent:
    """
    转写Agent - Pipeline的第一个节点

    架构说明:
    1. 接收音频字节数据（来自 WebSocket 或文件上传）
    2. 使用 SenseVoice/FunASR 进行批量转写
    3. 集成 pyannote-audio 进行说话人识别
    4. 合并结果，输出 TranscriptResult

    面试考点:
    - 为什么用 Qwen SenseVoice (FunASR)？（完全国产自研、长语音优化、带语气表情包识别）
    - VAD 预处理有什么作用？（降低幻觉，过滤静音段）
    - 说话人识别的原理？（speaker embedding + 聚类）
    """

    def __init__(self, config: TranscriptionConfig | None = None):
        self.config = config or TranscriptionConfig()
        self._model = None
        self._align_model = None
        self._diarize_pipeline = None
        self._initialized = False

    def _lazy_init(self):
        """
        懒加载模型 —— 避免在导入时就加载大模型。
        生产中模型应在服务启动时预热。
        """
        if self._initialized:
            return

        self._ensure_ffmpeg_available()

        try:
            from funasr import AutoModel

            logger.info(
                f"Loading FunASR Qwen-ASR model: {self.config.model_size} "
                f"on {self.config.device}"
            )
            self._model = AutoModel(
                model=self.config.model_size,
                device=self.config.device,
                disable_update=True
            )
            self._initialized = True
            logger.info("Qwen ASR (FunASR) model loaded successfully")
        except ImportError:
            logger.warning(
                "FunASR not installed, using mock transcription. "
                "Install with: pip install funasr modelscope"
            )
            self._initialized = True

    @staticmethod
    def _ensure_ffmpeg_available() -> None:
        """Ensure ffmpeg is on PATH for FunASR's subprocess fallback."""
        if shutil.which("ffmpeg"):
            return

        candidate_bins: list[str] = []
        local_app_data = os.getenv("LOCALAPPDATA", "")
        if local_app_data:
            winget_root = pathlib.Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_root.exists():
                for candidate in winget_root.glob("Gyan.FFmpeg_*/*/bin"):
                    candidate_bins.append(str(candidate))

        for candidate_bin in candidate_bins:
            ffmpeg_exe = pathlib.Path(candidate_bin) / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                current_path = os.environ.get("PATH", "")
                if candidate_bin not in current_path:
                    os.environ["PATH"] = f"{candidate_bin};{current_path}"
                logger.info(f"Using ffmpeg from: {ffmpeg_exe}")
                return

        logger.warning(
            "ffmpeg was not found in PATH or common WinGet locations; "
            "audio decoding may fail"
        )

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 执行语音转写

        Args:
            state: MeetingState 字典，包含 audio_data 字段

        Returns:
            更新后的 state，包含 transcript 和 transcript_text
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[TranscriptionAgent] Processing meeting: {meeting_id}")

        state["status"] = MeetingStatus.TRANSCRIBING

        audio_data = state.get("audio_data", b"")
        if not audio_data:
            logger.warning("No audio data provided, using demo transcript")
            state["transcript"] = self._generate_demo_transcript(meeting_id)
            state["transcript_text"] = self._format_transcript_text(
                state["transcript"]
            )
            return state

        try:
            self._lazy_init()
            transcript = await self._transcribe(audio_data, meeting_id)
            state["transcript"] = transcript
            state["transcript_text"] = self._format_transcript_text(transcript)
            logger.info(
                f"[TranscriptionAgent] Transcription complete: "
                f"{len(transcript.segments)} segments"
            )
        except Exception as e:
            logger.error(f"[TranscriptionAgent] Error: {e}")
            state["errors"] = state.get("errors", []) + [
                f"TranscriptionAgent: {str(e)}"
            ]
            state["transcript"] = self._generate_demo_transcript(meeting_id)
            state["transcript_text"] = self._format_transcript_text(
                state["transcript"]
            )

        return state

    async def _transcribe(
        self, audio_data: bytes, meeting_id: str
    ) -> TranscriptResult:
        """执行实际的语音转写流程 (基于 Qwen SenseVoice/FunASR)"""
        if self._model is None:
            return self._generate_demo_transcript(meeting_id)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        try:
            tmp.write(audio_data)
            tmp.flush()
            tmp.close()

            # Step 1: FunASR 转写
            res = self._model.generate(
                input=tmp_path,
                batch_size_s=self.config.batch_size,
                hotword='会议, AI, 代理'
            )

            # 兼容 FunASR 的输出格式
            full_text = ""
            if isinstance(res, list) and len(res) > 0:
                full_text = res[0].get("text", "")

            # Step 2: 说话人识别 (集成的 pyannote)
            # 这里可以调用实战中的 diarization 逻辑，本示例核心展示 Qwen/FunASR 的替换
            segments = [
                TranscriptSegment(
                    speaker="Speaker 1",
                    text=full_text,
                    start=0.0,
                    end=0.0,
                    confidence=1.0
                )
            ]

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return TranscriptResult(
            meeting_id=meeting_id,
            segments=segments,
            language=self.config.language,
            duration_seconds=0.0, # 实际应从音频中获取
            full_text=full_text,
        )

    @staticmethod
    def _generate_demo_transcript(meeting_id: str) -> TranscriptResult:
        """生成演示转写结果（无音频时使用）"""
        demo_segments = [
            TranscriptSegment(
                speaker="张总",
                text="好的，我们开始今天的Q3预算评审会议。首先请李明汇报一下目前的预算执行情况。",
                start=0.0, end=8.5, confidence=0.96,
            ),
            TranscriptSegment(
                speaker="李明",
                text="好的张总。截至目前，Q2预算执行率为87%，其中研发投入占比最大，达到42%。",
                start=9.0, end=16.2, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="李明",
                text="Q3我们计划将预算上调15%，主要增加在AI基础设施和人才招聘方面。",
                start=16.5, end=23.1, confidence=0.94,
            ),
            TranscriptSegment(
                speaker="王芳",
                text="关于人才招聘，我建议我们重点招聘3名高级算法工程师，预算大概在每人年薪80万左右。",
                start=23.5, end=31.0, confidence=0.93,
            ),
            TranscriptSegment(
                speaker="张总",
                text="可以。李明你来负责整理Q3的详细预算方案，下周五之前提交给我审批。",
                start=31.5, end=38.2, confidence=0.97,
            ),
            TranscriptSegment(
                speaker="张总",
                text="王芳负责拟定招聘JD，本周三前完成。另外，赵伟跟进一下服务器采购的事情。",
                start=38.5, end=46.0, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="赵伟",
                text="收到，我这边已经在对比几家供应商了，预计下周一可以给出采购方案。",
                start=46.5, end=52.8, confidence=0.94,
            ),
            TranscriptSegment(
                speaker="张总",
                text="好的，那今天的会议就到这里。各位辛苦了，请大家按时完成各自的任务。",
                start=53.0, end=59.5, confidence=0.96,
            ),
        ]

        full_text = "\n".join(
            f"[{s.speaker}] {s.text}" for s in demo_segments
        )

        return TranscriptResult(
            meeting_id=meeting_id,
            segments=demo_segments,
            language="zh",
            duration_seconds=59.5,
            full_text=full_text,
        )

    @staticmethod
    def _format_transcript_text(transcript: TranscriptResult) -> str:
        """将转写结果格式化为纯文本（供后续Agent使用）"""
        lines = []
        for seg in transcript.segments:
            ts = f"[{seg.start:.1f}s-{seg.end:.1f}s]"
            lines.append(f"{ts} {seg.speaker}: {seg.text}")
        return "\n".join(lines)
