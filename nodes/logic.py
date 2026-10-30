"""promboard.anime2d — 모델 없이 도는 판단(입력 검사 · 크기 맞추기 · 반복 프레임 · 채울 시각 · 쉬운 오류 문구).

torch 를 불러오지 않는다(numpy · Pillow 만). 시험은 저장소 tests/test_logic.py.
"""
import math

MODE_NAMES = {
    "inbetween": "두 프레임 → 중간 프레임",
    "tween": "두 프레임 → 영상",
    "smooth": "영상 → 프레임 늘리기",
}
# 모드 이름 뒤 조사(은 · 는) — 끝 글자 받침에 맞춘다.
_TOPIC = {"inbetween": "은", "tween": "은", "smooth": "는"}
PORT_NAMES = {"start": "시작 프레임", "end": "끝 프레임", "video": "늘릴 영상"}
# 모드마다 권하는 예제 카드(cards/cards.json 의 name 그대로)
EXAMPLE_CARDS = {
    "inbetween": ("예제 — 이어진 시작 프레임", "예제 — 이어진 끝 프레임"),
    "tween": ("예제 — 컬러 시작 프레임", "예제 — 컬러 끝 프레임"),
}
# 모드마다 필요한 그래픽카드 메모리(MB). manifest 모드 선택지 vramMinMb 와 같게 둔다(실측 2026-10-31).
MODE_VRAM_MB = {"inbetween": 4096, "tween": 10240, "smooth": 4096}


def _names(keys):
    return " · ".join(f"'{PORT_NAMES[k]}'" for k in keys)


def check_inputs(mode, inputs):
    """모드에 맞는 입력이 이어졌는지. 맞으면 None, 아니면 사용자에게 보일 한국어 한 문단(300자 안)."""
    name = f"'{MODE_NAMES[mode]}'{_TOPIC[mode]}"
    frames = [k for k in ("start", "end") if k in inputs]
    if mode in ("inbetween", "tween"):
        if "video" in inputs:
            return f"{name} '늘릴 영상'을 쓰지 않습니다. '늘릴 영상' 선을 빼거나 모드를 '영상 → 프레임 늘리기'로 바꾸세요."
        if len(frames) == 2:
            return None
        start_card, end_card = EXAMPLE_CARDS[mode]
        if not frames:
            now, cards = "지금은 아무것도 이어져 있지 않습니다.", f"'{start_card}' · '{end_card}'"
        else:
            now = f"지금은 {_names(frames)}만 이어져 있습니다."
            cards = f"'{end_card}'" if frames == ["start"] else f"'{start_card}'"
        return (
            f"{name} '시작 프레임'과 '끝 프레임'을 이어야 합니다. {now} "
            f"에셋의 {cards} 카드를 소스 노드 위에 끌어 놓아 보세요."
        )
    if "video" not in inputs:
        now = f"지금은 {_names(frames)}만 이어져 있습니다." if frames else "지금은 아무것도 이어져 있지 않습니다."
        return f"{name} '늘릴 영상'을 이어야 합니다. {now} 앞 노드의 영상 출력이나 소스 영상 노드를 '늘릴 영상'에 이어 보세요."
    if frames:
        return (
            f"{name} '늘릴 영상'만 씁니다. 지금은 {_names(frames)}도 이어져 있습니다. "
            "그 선을 빼거나 모드를 '두 프레임 → 중간 프레임'으로 바꾸세요."
        )
    return None


def device_problem(mode, device):
    """그래픽카드가 꼭 필요한 모드인데 CPU 로 돌게 됐으면 그 안내."""
    if mode == "tween" and device != "cuda":
        return (
            "'두 프레임 → 영상'은 NVIDIA 그래픽카드가 있어야 합니다. 앱 설정의 'CPU 만 쓰기'가 켜져 있으면 끄고, "
            "그래픽카드가 없는 PC 에서는 '두 프레임 → 중간 프레임'을 쓰세요."
        )
    return None


def length_problem(seconds, limit):
    """영상 길이 초과 안내. 받으면 None."""
    if seconds <= limit:
        return None
    return f"{limit}초까지 받습니다. 지금 영상은 {math.ceil(seconds)}초입니다."


def friendly_error(exc, mode, vram_mb):
    """그래픽카드 메모리 부족을 쉬운 말로. 다른 예외는 None(원래 예외를 그대로 둔다)."""
    text = str(exc)
    oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text.lower()
    if not oom:
        return None
    need = round(MODE_VRAM_MB[mode] / 1024)
    have = f"이 PC {round(vram_mb / 1024)} GB" if vram_mb else "이 PC 는 그래픽카드 메모리를 알 수 없음"
    tail = "다른 프로그램을 닫고 다시 하세요." if mode == "inbetween" else "다른 프로그램을 닫고 다시 하거나 '두 프레임 → 중간 프레임'을 쓰세요."
    return f"그래픽카드 메모리가 모자랍니다(이 모드 약 {need} GB 필요, {have}). {tail}"


# ── 크기 ────────────────────────────────────────────────────────────────


def match_size(img, size):
    """끝 프레임을 시작 프레임 크기에 맞춘다 — 비율이 다르면 가운데를 잘라 맞춘 뒤 줄인다(찌그러뜨리지 않는다)."""
    from PIL import Image

    if img.size == tuple(size):
        return img
    w, h = size
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    cw, ch = min(sw, round(w / scale)), min(sh, round(h / scale))
    left, top = (sw - cw) // 2, (sh - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def fit_frame(img, fit, size):
    """그림을 모델 크기(size = (가로, 세로))에 맞춘다. → (그림, 그림이 든 상자 (x0, y0, x1, y1)).

    pad: 다 보이게 줄이고 남는 곳은 그림 가장자리 평균 색으로 채운다(상자 = 그림이 든 곳, 결과를 이 상자로 다시 잘라 낸다).
    crop: 가운데를 잘라 꽉 채운다(상자 = 전체).
    """
    from PIL import Image

    img = img.convert("RGB")
    w, h = size
    sw, sh = img.size
    if fit == "crop":
        return match_size(img, size), (0, 0, w, h)
    scale = min(w / sw, h / sh)
    cw, ch = max(1, min(w, round(sw * scale))), max(1, min(h, round(sh * scale)))
    small = img.resize((cw, ch), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), _edge_color(small))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    canvas.paste(small, (x0, y0))
    return canvas, (x0, y0, x0 + cw, y0 + ch)


def _edge_color(img):
    import numpy as np

    a = np.asarray(img, dtype=np.float32)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    return tuple(int(round(v)) for v in edge.mean(axis=0))


def animeinterp_size(w, h, max_side=960):
    """AnimeInterp 처리 크기 — 긴 변이 max_side(학습 기준 960)를 넘으면 줄이고, 영상(yuv420p)이 받게 짝수로."""
    scale = min(1.0, max_side / max(w, h))
    return max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)


# ── 반복 프레임 · 채울 시각 ──────────────────────────────────────────────


# 같은 그림 판정: 긴 변 384 흑백에서 밝기가 16 넘게 바뀐 점이 전체의 0.02% 이하면 같은 그림.
# 실측(2026-10-31, 512×320 · 3프레임씩 반복 · H.264 crf 20 · 32): 같은 그림끼리 바뀐 점 0~12개, 다른 그림끼리 400개 넘게.
# 다른 그림을 같다고 보면 그 움직임을 건너뛰어 더 나쁘므로 같다고 보는 쪽을 좁게 잡았다(입모양처럼 아주 작은 변화는 같은 그림으로 볼 수 있다).
DEDUPE_THRESHOLD = 0.0002
_COMPARE_SIDE = 384
_CHANGED_LEVEL = 16


def _small_gray(frame):
    """비교용 작은 흑백(긴 변 384). numpy (h, w, 3) uint8 → int16 (h', w')."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(frame)
    scale = _COMPARE_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    return np.asarray(img.convert("L"), dtype=np.int16)


def dedupe_timeline(frames, threshold, fps):
    """반복된 프레임(한 그림을 2~3프레임씩 보여 주는 것)을 건너뛴 그림 목록 → [(프레임 번호, 시각 초)].

    바로 앞 그림과 견줘 밝기가 크게 바뀐 점의 비율이 threshold 이하면 같은 그림으로 본다. 첫 프레임은 늘 든다.
    """
    keys = []
    last = None
    for i, frame in enumerate(frames):
        g = _small_gray(frame)
        if last is None or float((abs(g - last) > _CHANGED_LEVEL).mean()) > threshold:
            keys.append((i, i / fps))
            last = g
    return keys


def fill_times(keys, n_frames, factor):
    """결과 프레임마다 (앞 그림 번호, 뒤 그림 번호, 사이 비율 t). t 가 0 이면 앞 그림을 그대로 쓴다.

    keys 는 그림이 바뀌는 원래 프레임 번호(오름차순, 첫 값 0). 결과는 n_frames × factor 장 — 길이 · 속도는 그대로다.
    시각은 결과 프레임 단위 정수로 센다(원래 i 번 = 결과 i × factor 번).
    """
    keys = list(keys) or [0]
    plan = []
    k = 0
    for j in range(n_frames * factor):
        while k + 1 < len(keys) and keys[k + 1] * factor <= j:
            k += 1
        a = keys[k]
        if k + 1 >= len(keys):
            plan.append((a, a, 0.0))
            continue
        b = keys[k + 1]
        t = (j - a * factor) / ((b - a) * factor)
        plan.append((a, a, 0.0) if t == 0 else (a, b, t))
    return plan


def pairs_of(plan):
    """만들어야 할 사이 프레임 — {(앞, 뒤): [t, …]}. 같은 쌍은 흐름을 한 번만 구한다."""
    out = {}
    for a, b, t in plan:
        if t > 0 and t not in out.setdefault((a, b), []):
            out[(a, b)].append(t)
    return out
