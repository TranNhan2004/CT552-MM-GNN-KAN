import time
import subprocess
import math

from pathlib import Path
from typing import List, Tuple


class UploadService:
    def __init__(self, upload_root_path: str) -> None:
        self.upload_root_path = Path(upload_root_path)

    def upload(self, files: List, upload_dir: str) -> Tuple[List[str], str, List[str]]:
        image_urls: List[str] = []
        audio_urls: List[str] = []
        text: str = ""

        full_path = self.upload_root_path / upload_dir
        full_path.mkdir(parents=True, exist_ok=True)

        for f in files:
            ext = f.filename.split(".")[-1].lower()
            save_path = full_path / f.filename

            if ext not in ("png", "jpg", "jpeg", "wav", "mp3", "txt", "mp4"):
                continue 

            save_path = self._save_file(f, save_path)

            if ext in ("png", "jpg", "jpeg"):
                image_urls.append(str(save_path))
            elif ext in ("wav", "mp3"):
                audio_urls.append(str(save_path))
            elif ext == "txt":
                with open(save_path, "r", encoding="utf-8") as txt_file:
                    content = txt_file.read().strip()
                    text += content if content else ""
            elif ext == "mp4":
                try:
                    duration = self._check_duration(save_path)
                    image_urls.extend(self._chunk_image(save_path, full_path))
                    audio_urls.extend(self._chunk_audio(save_path, full_path, duration))
                    self._clean(save_path)
                except Exception as e:
                    self._clean(full_path)
                    raise ValueError(f"Lỗi video {save_path}: {e}")

        return image_urls, text, audio_urls

    def _safe_save_path(self, file_path: Path) -> Path:
        timestamp = int(time.time() * 1000)  
        stem = file_path.stem
        suffix = file_path.suffix
        new_name = f"{stem}_{timestamp}{suffix}"
        return file_path.parent / new_name

    def _save_file(self, f, save_path: Path) -> Path:
        save_path = self._safe_save_path(save_path)
        f.file.seek(0)
        with open(save_path, "wb") as out_file:
            out_file.write(f.file.read())
        return save_path
    
    def _clean(self, path: Path) -> None:
        if not path.exists():
            return
        
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            for f in path.iterdir():
                if f.is_file():
                    f.unlink()
            path.rmdir()

    def _check_duration(self, video_path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        duration = float(result.stdout.strip())

        if duration < 30 or duration > 120:
            raise ValueError(f"Độ dài {duration:.1f}s không hợp lệ, phải từ 30s đến 2 phút")
        
        return duration

    def _chunk_image(self, video_path: Path, output_dir: Path) -> List[str]:
        image_urls = []
        out_pattern = str(output_dir / f"{video_path.stem}_image_%03d.jpg")
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=1/4",
            str(out_pattern),
            "-hide_banner", "-loglevel", "error", "-y"
        ], check=True)

        for img_file in sorted(output_dir.glob(f"{video_path.stem}_image_*.jpg")):
            image_urls.append(str(img_file))

        return image_urls

    def _chunk_audio(self, video_path: Path, output_dir: Path, duration: float) -> List[str]:
        audio_urls = []
        chunk_duration = 5
        n_chunks = math.ceil(duration / chunk_duration)

        for i in range(n_chunks):
            start_time = i * chunk_duration
            out_file = output_dir / f"{video_path.stem}_audio_{i+1:03d}.mp3"
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-ss", str(start_time),
                "-t", str(chunk_duration),
                "-vn", "-acodec", "mp3",
                str(out_file),
                "-hide_banner", "-loglevel", "error", "-y"
            ], check=True)
            audio_urls.append(str(out_file))

        return audio_urls