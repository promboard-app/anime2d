"""promboard.anime2d — 2D 애니메이션 · 프레임 사이 채우기 노드 코드 (ToonCrafter · AnimeInterp).

노드 하나(fill)가 모드 세 가지를 맡는다.
  두 프레임 → 중간 프레임 (inbetween, 처음 값)  AnimeInterp(CVPR21)로 시작 · 끝 프레임 사이 n장 + 셋을 이은 영상
  두 프레임 → 영상 (tween)                      ToonCrafter(512 보간판)로 16프레임 512×320 영상(8fps, 2초). NVIDIA 그래픽카드 필요
  영상 → 프레임 늘리기 (smooth)                 AnimeInterp 로 영상 fps 를 ×2 · ×3. 반복 프레임(한 그림을 2~3프레임씩 보여 주는
                                                2D 애니)은 건너뛰고 서로 다른 그림 사이를 원래 시각에 맞춰 채운다. 길이 · 속도 · 소리는 그대로

상류 코드는 nodes/tooncrafter(Doubiiu/ToonCrafter, Apache-2.0) · nodes/animeinterp(lisiyao21/AnimeInterp, MIT)에 최소로 고쳐 담았다
(고친 곳은 'PromBoard 고침' 주석, 목록은 NOTICE.md).
  - ToonCrafter: 공식 gradio 경로(scripts/gradio/i2v_test_application.py · scripts/evaluation/funcs.py)와 같은 순서 · 값
    (DDIM 50단계, cfg 7.5, eta 1.0, 시간 간격 uniform_trailing, guidance_rescale 0.7). 12GB 그래픽카드에 들게 바꾼 곳:
    모델 fp16(확산 일정 값은 fp32), 시작 · 끝 두 장만 인코딩, 단계마다 쓰는 덩어리만 그래픽카드에(_place), VAE 디코딩은
    autocast 없이 fp16 + 층마다 캐시 반환(_decode) — 실측 최고 6.3GB(예약 8.7GB). open_clip ViT-H-14 가중치는 model.ckpt 안에
    있어 따로 받지 않는다.
  - AnimeInterp: AnimeInterpNoCupy(순수 PyTorch 판)에 공식 체크포인트를 읽는다. 첫 흐름은 SGM 없이 0 에서 시작한다(설계 결정 —
    softsplat · torch_scatter 를 쓰지 않으려고). 흐름은 두 프레임마다 한 번 구하고 사이 비율마다 합성만 다시 한다.

노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import gc
import os
import random
import sys
import time
from contextlib import contextmanager

from . import logic

HERE = os.path.dirname(os.path.abspath(__file__))
TC_DIR = os.path.join(HERE, "tooncrafter")
TC_CKPT = "Doubiiu/ToonCrafter/model.ckpt"
AI_CKPT = "animeinterp/anime_interp_full.ckpt"
TC_SIZE = (512, 320)  # (가로, 세로) — 512 보간판이 학습한 크기
TC_FPS = 8  # 공식 save_fps
TC_STEPS = 50
TC_CFG = 7.5
TC_ETA = 1.0
TC_DEFAULT_PROMPT = "an anime scene"  # 공식 예시 글
# 움직임 크기 → frame_stride(공식 범위 5~30, 작을수록 움직임이 크다 — scripts/run.sh)
MOTION_STRIDE = {"small": 20, "normal": 10, "large": 5}
FACTORS = {"x2": 2, "x3": 3}
INBETWEEN_FPS = 12
# 늘릴 영상의 원래 프레임 수 상한(일이 프레임 수에 비례 — 24fps 로 그래픽카드 20초 · CPU 10초). 실측 2026-10-31 RTX 4070 SUPER:
# 1080p(960×540 로 처리) 24fps 20초 — ×2 423초 · ×3 약 744초, '그대로 두기' ×2 약 354초 · ×3 약 610초.
# CPU(같은 PC)는 512×320 에서 새 프레임 하나 약 1.8초.
SMOOTH_MAX_FRAMES = {"cuda": 480, "cpu": 240}
# PromBoard 실행기 노드 한 번 900초(CPU 는 ×4, 적재 몫 180초는 따로) — 노드를 시작한 때부터 세어 넘을 것 같으면 처음 8장 뒤 멈춘다.
TIME_BUDGET_SEC = {"cuda": 900, "cpu": 3600}

_CACHE = {"kind": None, "model": None}


# ── 진입 ────────────────────────────────────────────────────────────────


def fill(ctx, inputs, settings):
    mode = settings.get("mode", "inbetween")
    problem = logic.check_inputs(mode, inputs) or logic.device_problem(mode, ctx.device)
    if problem:
        raise ValueError(problem)
    try:
        if mode == "tween":
            return _tween(ctx, inputs, settings)
        if mode == "smooth":
            return _smooth(ctx, inputs, settings)
        return _inbetween(ctx, inputs, settings)
    except Exception as exc:  # noqa: BLE001 - 그래픽카드 메모리 부족만 쉬운 말로 바꾸고 나머지는 그대로 올린다
        msg = logic.friendly_error(exc, mode, _vram_mb())
        if msg is None:
            raise
        _free()
        raise RuntimeError(msg) from None


NODES = {"fill": fill}


# ── 공통 ────────────────────────────────────────────────────────────────


def _vram_mb():
    import torch

    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024) if torch.cuda.is_available() else 0


def _free():
    """앞 모드의 모델을 내린다(모드가 바뀌거나 메모리가 모자랄 때)."""
    import torch

    _CACHE["kind"] = None
    _CACHE["model"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed(value):
    import numpy as np
    import torch

    seed = random.randint(0, 2**31 - 1) if value is None or int(value) < 0 else int(value)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _open_image(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))  # 투명한 곳은 흰 바탕(선화 PNG)
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB")


def _even_box(box):
    """yuv420p 영상은 짝수 크기만 받는다 — 상자 오른쪽 · 아래를 1px 줄여 맞춘다."""
    x0, y0, x1, y1 = box
    return x0, y0, x1 - (x1 - x0) % 2, y1 - (y1 - y0) % 2


def _save_frames(ctx, arrays, prefix):
    from PIL import Image

    paths = []
    for i, arr in enumerate(arrays):
        p = ctx.out_path(f"{prefix}-{i:02d}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


class _VideoWriter:
    """H.264(libx264) · yuv420p mp4. 프레임마다 번호(pts)를 0, 1, 2 … 로 붙인다(PromBoard YOLO 플러그인과 같은 꼴)."""

    def __init__(self, path, width, height, rate, audio_from=None, ctx=None):
        from fractions import Fraction

        import av

        rate = Fraction(rate).limit_denominator(1001000)  # PyAV 는 time_base 에 분수만 받는다
        self.container = av.open(path, "w", options={"movflags": "+faststart"})
        stream = self.container.add_stream("libx264", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "17", "preset": "medium"}
        stream.codec_context.time_base = 1 / rate
        self.stream = stream
        self.time_base = 1 / rate
        self.count = 0
        self.ctx = ctx
        self.audio = None
        if audio_from is not None:
            src, astream = audio_from
            if astream.codec_context.name in ("aac", "mp3"):
                self.audio = (src, astream, self.container.add_stream_from_template(astream), True)
            else:
                self.audio = (src, astream, self.container.add_stream("aac", rate=astream.rate or 48000), False)

    def write(self, array):
        import av
        import numpy as np

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
        frame.pts = self.count
        frame.time_base = self.time_base
        self.count += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        try:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
            if self.audio is not None:
                self._copy_audio()
        finally:
            self.container.close()

    def _copy_audio(self):
        """원본 소리를 옮긴다. aac · mp3 는 패킷 그대로, 그 밖은 aac 로 다시 담는다(PromBoard SeedVR2 플러그인과 같은 꼴)."""
        import av

        src, astream, aout, copy = self.audio
        try:
            src.seek(0)
            if copy:
                for pkt in src.demux(astream):
                    if pkt.dts is None:
                        continue
                    pkt.stream = aout
                    self.container.mux(pkt)
            else:
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=aout.rate)
                for frame in src.decode(astream):
                    for rf in resampler.resample(frame):
                        for pkt in aout.encode(rf):
                            self.container.mux(pkt)
                for pkt in aout.encode(None):
                    self.container.mux(pkt)
        except Exception as exc:  # noqa: BLE001 - 소리를 못 옮겨도 영상은 낸다
            if self.ctx is not None:
                self.ctx.log(f"audio copy failed: {exc}", "warn")


def _write_video(ctx, name, arrays, rate):
    h, w = arrays[0].shape[:2]
    path = ctx.out_path(name)
    writer = _VideoWriter(path, w, h, rate)
    for arr in arrays:
        writer.write(arr)
    writer.close()
    return path


@contextmanager
def _no_init():
    """모델 틀을 만드는 동안 가중치 무작위 초기화를 건너뛴다(곧 체크포인트로 덮으니 — 2.6B 개 초기화 시간과 메모리를 아낀다)."""
    import torch

    names = ["uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_",
             "zeros_", "ones_", "constant_", "orthogonal_"]
    saved = {n: getattr(torch.nn.init, n) for n in names}
    for n in names:
        setattr(torch.nn.init, n, lambda tensor, *a, **k: tensor)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


# ── AnimeInterp ─────────────────────────────────────────────────────────


def _animeinterp(ctx):
    if _CACHE["kind"] == "animeinterp":
        return _CACHE["model"]
    _free()
    import torch

    from .animeinterp.models import AnimeInterpNoCupy

    ctx.progress(0.0, "load-model")
    model = AnimeInterpNoCupy(path=None)
    # 공식 체크포인트는 torch 1.x 옛 저장 형식(zip 아님)이라 mmap 으로 열 수 없다 — 24.9MB 라 그냥 읽는다.
    state = torch.load(ctx.model_path(AI_CKPT), map_location="cpu", weights_only=True, mmap=False)
    state = state.get("model_state_dict", state)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}  # DataParallel 'module.' 떼기
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"AnimeInterp checkpoint is missing {len(missing)} weights (e.g. {missing[0]})")
    if unexpected:
        ctx.log(f"animeinterp: {len(unexpected)} unused checkpoint keys (e.g. {unexpected[0]})")
    model.eval().to(ctx.device)
    _CACHE.update(kind="animeinterp", model=model)
    return model


def _to_tensor(arr, device):
    """(h, w, 3) uint8 → (1, 3, H, W) float 0~1. logic.flow_pad 크기(8의 배수, 64 이상)로 가장자리 복제 덧대기
    (흐름 1/8 · 합성 1/4 크기가 맞고, 64px 보다 작은 그림도 받게 — 결과는 _Interp.at 이 원래 크기로 잘라 낸다)."""
    import torch
    import torch.nn.functional as F

    t = torch.tensor(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0).contiguous().to(device)  # 앞으로 밀기는 view(-1) 를 쓴다
    h, w = arr.shape[:2]
    pw, ph = logic.flow_pad(w, h)
    if (pw, ph) != (w, h):
        t = F.pad(t, (0, pw - w, 0, ph - h), mode="replicate")
    return t


class _Interp:
    """두 프레임의 흐름을 한 번 구해 두고 사이 비율 t 마다 합성한다. 공식 AnimeInterpNoCupy.forward 를 둘로 나눈 것이다."""

    def __init__(self, model, a, b, device):
        import torch

        self.model, self.h, self.w = model, a.shape[0], a.shape[1]
        with torch.no_grad():
            self.I1, self.I2 = _to_tensor(a, device), _to_tensor(b, device)
            I1o, I2o = (self.I1 - 0.5) / 0.5, (self.I2 - 0.5) / 0.5
            self.f1 = model.feat_ext(I1o)
            self.f2 = model.feat_ext(I2o)
            # SGM 첫 흐름 없이(flow_init=None — 0 에서 시작) 공식 12번 다듬기
            self.F12, _, _ = model.flownet(I1o, I2o, iters=12, test_mode=False, flow_init=None)
            self.F21, _, _ = model.flownet(I2o, I1o, iters=12, test_mode=False, flow_init=None)

    def at(self, t):
        """사이 비율 t(0~1) 프레임 → (h, w, 3) uint8."""
        import torch

        m = self.model
        with torch.no_grad():
            F1t, F2t = t * self.F12, (1 - t) * self.F21
            (f11, f12, f13), (f21, f22, f23) = self.f1, self.f2
            I1t, n1 = m.fwarp(self.I1, F1t)
            I2t, n2 = m.fwarp(self.I2, F2t)
            w1 = [m.fwarp(f, m.dflow(F1t, f)) for f in (f11, f12, f13)]
            w2 = [m.fwarp(f, m.dflow(F2t, f)) for f in (f21, f22, f23)]
            for img, norm in [(I1t, n1), (I2t, n2)] + w1 + w2:
                img[norm > 0] = img.clone()[norm > 0] / norm[norm > 0]
            out = m.synnet(
                torch.cat([I1t, I2t], dim=1),
                torch.cat([w1[0][0], w2[0][0]], dim=1),
                torch.cat([w1[1][0], w2[1][0]], dim=1),
                torch.cat([w1[2][0], w2[2][0]], dim=1),
            )
            out = out[0, :, : self.h, : self.w].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0)
            return out.cpu().numpy()


def _resize_np(arr, size):
    import numpy as np
    from PIL import Image

    if (arr.shape[1], arr.shape[0]) == tuple(size):
        return arr
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def _inbetween(ctx, inputs, settings):
    import numpy as np

    count = int(settings.get("count", 7))
    start = _open_image(ctx.in_path(inputs["start"]))
    end = logic.match_size(_open_image(ctx.in_path(inputs["end"])), start.size)
    size = logic.animeinterp_size(*start.size)
    a = _resize_np(np.asarray(start), size)
    b = _resize_np(np.asarray(end), size)
    ctx.log(f"inbetween {start.size[0]}x{start.size[1]} -> {size[0]}x{size[1]}, {count} frames, device {ctx.device}")
    model = _animeinterp(ctx)
    ctx.progress(0.05, "flow")
    interp = _Interp(model, a, b, ctx.device)
    mids = []
    for k in range(1, count + 1):
        ctx.check_cancel()
        mids.append(interp.at(k / (count + 1)))
        ctx.progress(0.1 + 0.8 * k / count, "work")
    frames = _save_frames(ctx, mids, "between")
    video = _write_video(ctx, "inbetween.mp4", [a] + mids + [b], INBETWEEN_FPS)
    ctx.progress(1.0, "done")
    return {"video": video, "frames": frames}


# ── 영상 → 프레임 늘리기 ─────────────────────────────────────────────────


def _read_video(ctx, path, max_frames):
    """영상 → (RGB 프레임 목록 (h, w, 3) uint8 — AnimeInterp 처리 크기, fps(Fraction), 원본 컨테이너 · 소리 스트림).
    원래 프레임이 max_frames 를 넘으면(0.5초 여유) 한국어 오류."""
    import av
    import numpy as np
    from PIL import Image

    src = av.open(path)
    if not src.streams.video:
        src.close()
        raise ValueError("'늘릴 영상'에 영상이 없습니다.")
    stream = src.streams.video[0]
    rate = stream.average_rate or stream.guessed_rate
    if not rate or float(rate) <= 0:
        src.close()
        raise ValueError("'늘릴 영상'의 fps 를 읽지 못했습니다.")
    fps = float(rate)
    seconds = float(stream.frames / rate) if stream.frames else (float(stream.duration * stream.time_base) if stream.duration else 0.0)
    problem = logic.length_problem(seconds, fps, max_frames)
    if problem:
        src.close()
        raise ValueError(problem)
    frames, size = [], None
    try:
        for frame in src.decode(stream):
            ctx.check_cancel()
            arr = frame.to_ndarray(format="rgb24")
            rot = int(round((getattr(frame, "rotation", 0) or 0) / 90.0)) % 4
            if rot:
                arr = np.ascontiguousarray(np.rot90(arr, rot))
            if size is None:
                size = logic.animeinterp_size(arr.shape[1], arr.shape[0])
            if (arr.shape[1], arr.shape[0]) != size:
                arr = np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))
            frames.append(arr)
            problem = logic.length_problem(len(frames) / fps, fps, max_frames, at_least=True)  # 머리 정보의 길이가 틀린 영상
            if problem:
                raise ValueError(problem)
    except av.error.FFmpegError as exc:  # 끝부분이 깨진 파일 — 읽은 데까지 쓴다
        ctx.log(f"video decode stopped: {exc}", "warn")
    except Exception:
        src.close()
        raise
    if not frames:
        src.close()
        raise ValueError("'늘릴 영상'에서 프레임을 하나도 읽지 못했습니다.")
    astream = src.streams.audio[0] if src.streams.audio else None
    return frames, rate, src, astream


def _smooth(ctx, inputs, settings):
    import torch

    began = time.monotonic()  # 노드 시간 상한은 영상 읽기 · 반복 찾기 · 모델 적재까지 센다
    factor = FACTORS.get(settings.get("factor", "x2"), 2)
    dev = "cuda" if ctx.device == "cuda" else "cpu"
    max_frames, budget = SMOOTH_MAX_FRAMES[dev], TIME_BUDGET_SEC[dev]
    ctx.progress(0.0, "read")
    frames, rate, src, astream = _read_video(ctx, ctx.in_path(inputs["video"]), max_frames)
    try:
        n = len(frames)
        if settings.get("repeats", "skip") == "keep":
            keys = list(range(n))
        else:
            keys = [i for i, _ in logic.dedupe_timeline(frames, logic.DEDUPE_THRESHOLD, float(rate), ctx.check_cancel)]
        plan = logic.fill_times(keys, n, factor)
        pairs = logic.pairs_of(plan)
        made = sum(len(ts) for ts in pairs.values())
        h, w = frames[0].shape[:2]
        ctx.log(f"smooth {n} frames {w}x{h} {float(rate):.3f}fps, {len(keys)} drawings, x{factor} -> {len(plan)} frames ({made} new)")
        model = _animeinterp(ctx)
        path = ctx.out_path("smooth.mp4")
        writer = _VideoWriter(path, w, h, rate * factor, (src, astream) if astream is not None else None, ctx)
        current, interp = None, None
        done, written, new_sec, write_sec = 0, 0, 0.0, 0.0  # 새 프레임 수 · 쓴 프레임 수와 각각에 든 시간(어림용)
        try:
            for a, b, t in plan:
                ctx.check_cancel()
                if t > 0:
                    t0 = time.monotonic()
                    if current != (a, b):
                        interp = None
                        if dev == "cuda":
                            torch.cuda.empty_cache()  # 쌍마다 새 흐름 — 캐시가 조각나 불어나지 않게
                        current, interp = (a, b), _Interp(model, frames[a], frames[b], ctx.device)
                    out = interp.at(t)
                    new_sec += time.monotonic() - t0
                    done += 1
                else:
                    out = frames[a]
                t0 = time.monotonic()
                writer.write(out)
                write_sec += time.monotonic() - t0
                written += 1
                if t == 0:
                    continue
                ctx.progress(0.05 + 0.9 * done / max(1, made), "work")
                if done == 8 and made > 16:
                    spent = time.monotonic() - began
                    guess = logic.smooth_estimate(spent, done, new_sec, made, written, write_sec, len(plan))
                    ctx.log(
                        f"first {done} new frames {new_sec:.1f}s, {written} written {write_sec:.1f}s, "
                        f"{spent:.1f}s since start, whole about {guess:.0f}s (budget {budget}s)"
                    )
                    problem = logic.time_problem(guess, budget, factor)
                    if problem:
                        raise ValueError(problem)
        finally:
            writer.close()
    finally:
        src.close()
    ctx.progress(1.0, "done")
    return {"video": path}


# ── ToonCrafter ─────────────────────────────────────────────────────────


def _tooncrafter(ctx):
    if _CACHE["kind"] == "tooncrafter":
        return _CACHE["model"]
    _free()
    import torch

    if TC_DIR not in sys.path:
        sys.path.insert(0, TC_DIR)  # 상류 코드가 lvdm · utils 를 맨 위 이름으로 부르고, 설정 파일 target 도 그 이름이다
    from omegaconf import OmegaConf
    from utils.utils import instantiate_from_config

    config = OmegaConf.load(os.path.join(TC_DIR, "configs", "inference_512_v1.0.yaml"))
    mc = config.pop("model", OmegaConf.create())
    mc["params"]["unet_config"]["params"]["use_checkpoint"] = False  # 공식 gradio 와 같다
    # open_clip ViT-H-14 가중치는 model.ckpt 안에 있다 — 따로 받지 않는다(실행 중 인터넷 없음).
    mc["params"]["cond_stage_config"]["params"]["version"] = None
    mc["params"]["img_cond_stage_config"]["params"]["version"] = None
    ctx.progress(0.0, "load-model")
    with _no_init():
        model = instantiate_from_config(mc)
    state = torch.load(ctx.model_path(TC_CKPT), map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    ctx.progress(0.0, "load-model")
    # 큰 덩어리만 fp16 으로(12GB 그래픽카드에 들게). 확산 일정 값(alphas_cumprod 등)은 fp32 그대로 둔다.
    for part in (model.model, model.first_stage_model, model.cond_stage_model, model.embedder, model.image_proj_model):
        part.half()
    from lvdm.basics import GroupNormSpecific

    for m in model.first_stage_model.modules():  # 상류 GroupNormSpecific 는 fp32 로 정규화한다(x.float()) — 가중치도 fp32 로 둔다(_decode)
        if isinstance(m, GroupNormSpecific):
            m.float()
    model = model.eval().to(ctx.device)
    model.cond_stage_model.device = ctx.device
    model.embedder.device = ctx.device
    _place(model, ctx.device, set())
    _CACHE.update(kind="tooncrafter", model=model)
    return model


def _place(model, device, on_gpu):
    """단계마다 그 단계가 쓰는 덩어리만 그래픽카드에 둔다 — 글 · 그림 인코더(clip, 약 2GB)는 조건 만들 때, 확산 모델(unet, 약 2.8GB)은
    샘플링 때만. 셋을 다 올리면 16프레임 디코딩(VAE)이 12GB 를 넘는다(실측 14.3GB)."""
    import torch

    for name, parts in (("clip", (model.cond_stage_model, model.embedder)), ("unet", (model.model,))):
        for part in parts:
            part.to(device if name in on_gpu else "cpu")
    torch.cuda.empty_cache()


class _MemLog:
    """단계마다 그래픽카드 메모리 최고치를 로그에 남긴다(메모리 부족 진단용)."""

    def __init__(self, ctx):
        import torch

        self.ctx, self.cuda = ctx, torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def __call__(self, stage):
        import torch

        if self.cuda:
            mb = 1024 * 1024
            self.ctx.log(
                f"vram {stage}: peak {torch.cuda.max_memory_allocated() // mb}MB (reserved {torch.cuda.max_memory_reserved() // mb}MB), "
                f"now {torch.cuda.memory_allocated() // mb}MB"
            )
            torch.cuda.reset_peak_memory_stats()


def _tc_tensor(img, device):
    import numpy as np
    import torch

    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().to(device)
    return (t / 255.0 - 0.5) * 2  # 공식과 같이 -1~1


def _tween(ctx, inputs, settings):
    import numpy as np
    import torch
    from einops import rearrange

    start = _open_image(ctx.in_path(inputs["start"]))
    end = logic.match_size(_open_image(ctx.in_path(inputs["end"])), start.size)
    fit = settings.get("fit", "pad")
    a, box = logic.fit_frame(start, fit, TC_SIZE)
    b, _ = logic.fit_frame(end, fit, TC_SIZE)
    prompt = (inputs.get("prompt") or "").strip() or TC_DEFAULT_PROMPT
    stride = MOTION_STRIDE.get(settings.get("motion", "normal"), 10)
    model = _tooncrafter(ctx)
    # 모델 틀을 처음 만들 때 난수를 쓴다(open_clip 등) — 시드는 그 뒤에 정해야 첫 실행도 같은 시드에 같은 결과가 난다.
    seed = _seed(settings.get("seed", -1))
    ctx.log(f"tween {start.size[0]}x{start.size[1]} fit {fit} box {box}, frame_stride {stride}, seed {seed}")
    from lvdm.models.samplers.ddim import DDIMSampler

    ctx.progress(0.02, "encode")
    device = ctx.device
    mem = _MemLog(ctx)
    frames = model.temporal_length
    h, w = TC_SIZE[1] // 8, TC_SIZE[0] // 8
    noise_shape = [1, model.model.diffusion_model.out_channels, frames, h, w]
    started = time.monotonic()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        # 공식 Image2Video.get_image 와 같은 순서
        _place(model, device, {"clip"})
        text_emb = model.get_learned_conditioning([prompt])
        ta, tb = _tc_tensor(a, device), _tc_tensor(b, device)
        z, hs = _latent_with_hidden_states(model, ta, tb)
        img_tensor_repeat = torch.zeros(1, z.shape[1], frames, z.shape[3], z.shape[4], device=device, dtype=z.dtype)
        img_tensor_repeat[:, :, :1] = z[:, :, :1]
        img_tensor_repeat[:, :, -1:] = z[:, :, -1:]
        img_emb = model.image_proj_model(model.embedder(ta.unsqueeze(0)))
        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        fs = torch.tensor([stride], dtype=torch.long, device=device)
        cond = {"c_crossattn": [imtext_cond], "c_concat": [img_tensor_repeat]}  # fs 는 따로 넘긴다(공식은 cond 에서 지운다)
        # 무조건 조건(빈 글 + 빈 그림) — 공식 batch_ddim_sampling
        uc_emb = model.get_learned_conditioning([""])
        uc_img = model.image_proj_model(model.embedder(torch.zeros(1, 3, 224, 224, device=device)))
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img], dim=1)], "c_concat": [img_tensor_repeat]}
        mem("conditioning")
        _place(model, device, {"unet"})
        sampler = DDIMSampler(model)
        step = {"n": 0}

        def on_step(_i):
            step["n"] += 1
            ctx.check_cancel()
            ctx.progress(0.05 + 0.8 * step["n"] / TC_STEPS, "work")

        samples, _ = sampler.sample(
            S=TC_STEPS, conditioning=cond, batch_size=1, shape=noise_shape[1:], verbose=False,
            unconditional_guidance_scale=TC_CFG, unconditional_conditioning=uc, eta=TC_ETA,
            temporal_length=frames, conditional_guidance_scale_temporal=None, x_T=None, fs=fs,
            timestep_spacing="uniform_trailing", guidance_rescale=0.7, clean_cond=True, callback=on_step,
        )
        mem("sampling")
        _place(model, device, set())
        ctx.progress(0.86, "decode")
        video = _decode(model, samples, hs)
        torch.cuda.empty_cache()
        # 공식: 두 번째 · 끝에서 두 번째를 뺀 14장을 다시 풀어 가운데 두 장을 바꾼다(funcs.batch_ddim_sampling)
        index = list(range(samples.shape[2]))
        del index[1]
        del index[-2]
        middle = _decode(model, samples[:, :, index], hs)
        half = video.shape[2] // 2
        video[:, :, half - 1 : half + 1] = middle[:, :, half - 2 : half]
        mem("decode")
    ctx.log(f"tween sampled and decoded in {time.monotonic() - started:.1f}s")
    arr = ((video[0].float().clamp(-1, 1) + 1) / 2 * 255).round().byte()  # c t h w
    arr = rearrange(arr, "c t h w -> t h w c").cpu().numpy()
    x0, y0, x1, y1 = _even_box(box)
    arr = [np.ascontiguousarray(f[y0:y1, x0:x1]) for f in arr]
    del samples, video, middle
    frames_out = _save_frames(ctx, arr, "tween")
    path = _write_video(ctx, "tween.mp4", arr, TC_FPS)
    ctx.progress(1.0, "done")
    return {"video": path, "frames": frames_out}


def _decode(model, samples, hs):
    """공식 decode_first_stage(decode_core) 와 같은 계산을 autocast 없이 fp16 으로 — autocast 는 GroupNorm 결과를 fp32 로 두어
    16프레임 512×320 디코딩이 9.5GB 를 쓴다. 참조(ref_context)를 넘기면 공식도 post_quant_conv 를 건너뛴다(같은 kwargs 를 넘긴다)."""
    import torch
    from einops import rearrange

    b, _, t, _, _ = samples.shape
    z = rearrange(samples, "b c t h w -> (b t) c h w")
    z = (1.0 / model.scale_factor * z).half()
    # 디코더 층마다 쓰고 남은 캐시를 돌려준다 — 윈도우 드라이버는 그래픽카드 메모리가 차면 오류 대신 컴퓨터 메모리로 넘겨
    # (sysmem fallback) 캐시가 쓰는 양(약 6GB)의 두 배 가까이 불어난다.
    blocks = [m for m in model.first_stage_model.decoder.modules() if not list(m.children())]
    hooks = [m.register_forward_hook(lambda *_: torch.cuda.empty_cache()) for m in blocks]
    try:
        with torch.autocast("cuda", enabled=False):
            out = model.first_stage_model.decode(z, ref_context=[h.half() for h in hs], timesteps=t)
    finally:
        for hk in hooks:
            hk.remove()
    return rearrange(out, "(b t) c h w -> b c t h w", b=b, t=t)


def _latent_with_hidden_states(model, ta, tb):
    """공식 Image2Video.get_latent_z_with_hidden_states 와 같은 값 — 첫 · 끝 프레임의 잠재값과 인코더 중간값(디코더 참조).

    공식은 시작 프레임 8장 + 끝 프레임 8장(16장)을 통째로 인코딩한 뒤 첫 · 끝 것만 쓴다. 인코더는 프레임마다 따로 돌므로 두 장만
    인코딩해도 값이 같다(잠재값 표본의 난수 순서만 다르다) — 16장 인코딩이 10GB 를 넘게 써서 바꿨다."""
    import torch
    from einops import rearrange

    x = torch.stack([ta, tb])
    encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)
    first_last = [rearrange(hid, "(b t) c h w -> b c t h w", t=2) for hid in hidden_states]
    z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = rearrange(z, "(b t) c h w -> b c t h w", b=1, t=2)
    return z, first_last
