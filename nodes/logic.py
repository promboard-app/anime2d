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
# 앱 설정 이름(PromBoard '로컬 AI 모델' 창 · locales/ko/localai.json behavior.forceCpu). 플러그인 노드도 이 설정을 따른다.
CPU_ONLY_SETTING = "'로컬 AI 모델' 설정의 '그래픽카드 대신 CPU 만 쓰기'"


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
            f"'두 프레임 → 영상'은 NVIDIA 그래픽카드가 있어야 합니다. {CPU_ONLY_SETTING}가 켜져 있으면 끄고, "
            "그래픽카드가 없는 PC 에서는 '두 프레임 → 중간 프레임'을 쓰세요."
        )
    return None


# 경계에 걸친 영상을 받는 여유(초) — 23.976fps 20초는 480프레임 = 20.02초다.
LENGTH_SLACK_SEC = 0.5


def _num(x, digits=1):
    """소수 digits 자리까지, 끝의 0 은 뺀다 — 20.0 → '20', 20.62 → '20.6', 23.976 (2자리) → '23.98'."""
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def length_problem(seconds, fps, max_frames, at_least=False):
    """영상 길이 초과 안내. 받으면 None.

    상한은 원래 프레임 수(max_frames)다 — 일이 프레임 수에 비례하므로 fps 가 높으면 받는 초가 그만큼 짧다(초로 바꿔 보인다).
    at_least: 머리 정보에 길이가 없어 읽다가 넘은 것(정확한 길이를 모른다).
    """
    limit = max_frames / fps
    if seconds <= limit + LENGTH_SLACK_SEC:
        return None
    now = f"{_num(limit)}초보다 깁니다" if at_least else f"{_num(seconds)}초입니다"
    return (
        f"이 영상({_num(fps, 2)}fps)은 {_num(limit)}초까지 받습니다(한 번에 원래 프레임 {max_frames}장까지). "
        f"지금 영상은 {now}. 영상을 짧게 잘라 주세요."
    )


# ── 프레임 늘리기 시간 어림 ──────────────────────────────────────────────


# 노드 한 번 상한의 90% 를 넘을 것 같으면 멈춘다(끝내기 직전에 끊기지 않게).
TIME_MARGIN = 0.9
# 인코더(libx264 preset medium)가 끝날 때 한꺼번에 내보내는 프레임 몫 — 앞을 미리 보느라 쌓아 둔다.
ENCODE_TAIL_FRAMES = 60


def smooth_estimate(spent, new_done, new_sec, new_total, written, write_sec, out_total):
    """프레임 늘리기 전체 시간 어림(초).

    지금까지 쓴 시간(영상 읽기 · 반복 찾기 · 모델 적재 포함) + 남은 새 프레임 × 한 장 만들기 시간
    + (남은 쓰기 + 인코더 마무리 몫) × 한 장 쓰기(인코딩) 시간.
    new_sec · write_sec 는 지금까지 만들기 · 쓰기에 든 시간의 합이다.
    """
    per_new = new_sec / max(1, new_done)
    per_write = write_sec / max(1, written)
    return spent + max(0, new_total - new_done) * per_new + (max(0, out_total - written) + ENCODE_TAIL_FRAMES) * per_write


def time_problem(guess, budget, factor):
    """어림 시간이 상한의 90% 를 넘으면 그 안내. 아니면 None."""
    if guess <= budget * TIME_MARGIN:
        return None
    tip = "영상을 짧게 자르거나 배수를 '×2'로 낮춰 주세요." if factor > 2 else "영상을 짧게 잘라 주세요."
    return (
        f"이 영상은 이 PC 에서 약 {math.ceil(guess / 60)}분 걸릴 것 같아 멈췄습니다. "
        f"노드 한 번은 {budget // 60}분까지라 끝내기 전에 끊길 수 있습니다. {tip}"
    )


def friendly_error(exc, mode, vram_mb):
    """그래픽카드 메모리 부족을 쉬운 말로. 다른 예외는 None(원래 예외를 그대로 둔다).

    AnimeInterp 모드(중간 프레임 · 프레임 늘리기)는 두 프레임씩 차례로 처리해 영상 길이 · 배수와 상관없이 메모리가 같다 —
    그래픽카드를 비우거나 CPU 로 돌리는 것만 듣는다.
    """
    text = str(exc)
    oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text.lower()
    if not oom:
        return None
    need = round(MODE_VRAM_MB[mode] / 1024)
    have = f"이 PC {round(vram_mb / 1024)} GB" if vram_mb else "이 PC 는 그래픽카드 메모리를 알 수 없음"
    if mode == "tween":
        tail = "다른 프로그램을 닫고 다시 하거나 '두 프레임 → 중간 프레임'을 쓰세요."
    else:
        tail = f"그래픽카드를 쓰는 다른 프로그램을 닫고 다시 하거나, {CPU_ONLY_SETTING}를 켜고 다시 하세요(느립니다)."
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


# AnimeInterp 흐름(RAFT 상관 피라미드)은 1/8 크기에서 4단을 반씩 줄인다 — 가로나 세로가 64px 보다 작으면 avg_pool 이 영어 오류를 낸다.
FLOW_MIN_SIDE = 64


def flow_pad(w, h):
    """흐름 · 합성에 넣을 크기 (가로, 세로) — 8의 배수로 올리고 64 보다 작으면 64. 가장자리를 복제해 덧대고 결과는 원래 크기로 잘라 낸다."""

    def up(v):
        return max(FLOW_MIN_SIDE, -(-v // 8) * 8)

    return up(w), up(h)


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


def dedupe_timeline(frames, threshold, fps, check=None):
    """반복된 프레임(한 그림을 2~3프레임씩 보여 주는 것)을 건너뛴 그림 목록 → [(프레임 번호, 시각 초)].

    바로 앞 그림과 견줘 밝기가 크게 바뀐 점의 비율이 threshold 이하면 같은 그림으로 본다. 첫 프레임은 늘 든다.
    check: 프레임마다 부른다(노드의 ctx.check_cancel — 취소를 바로 받게).
    """
    keys = []
    last = None
    for i, frame in enumerate(frames):
        if check is not None:
            check()
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
