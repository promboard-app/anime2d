"""모델 없이 도는 순수 함수 시험(pytest). 저장소 루트에서 `python -m pytest tests` — 설치 zip 에는 들지 않는다(.gitattributes)."""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nodes import logic  # noqa: E402


# ── 입력 검사 (모델을 불러오기 전에 한국어 오류) ─────────────────────────


def test_tween_needs_both_frames():
    msg = logic.check_inputs("tween", {"start": "a.png"})
    assert msg == (
        "'두 프레임 → 영상'은 '시작 프레임'과 '끝 프레임'을 이어야 합니다. 지금은 '시작 프레임'만 이어져 있습니다. "
        "에셋의 '예제 — 컬러 끝 프레임' 카드를 소스 노드 위에 끌어 놓아 보세요."
    )


def test_end_only_is_an_error_and_names_the_mode_card():
    msg = logic.check_inputs("inbetween", {"end": "b.png"})
    assert "지금은 '끝 프레임'만 이어져 있습니다" in msg
    assert "'예제 — 이어진 시작 프레임'" in msg
    assert msg.startswith("'두 프레임 → 중간 프레임'은 ")


def test_nothing_connected_names_both_cards():
    msg = logic.check_inputs("tween", {})
    assert "지금은 아무것도 이어져 있지 않습니다" in msg
    assert "'예제 — 컬러 시작 프레임' · '예제 — 컬러 끝 프레임'" in msg


def test_video_with_two_frame_mode_is_an_error():
    msg = logic.check_inputs("tween", {"start": "a.png", "end": "b.png", "video": "v.mp4"})
    assert "'늘릴 영상'을 쓰지 않습니다" in msg
    assert "'영상 → 프레임 늘리기'" in msg


def test_smooth_needs_video_and_says_what_is_connected():
    msg = logic.check_inputs("smooth", {"start": "a.png"})
    assert msg.startswith("'영상 → 프레임 늘리기'는 '늘릴 영상'을 이어야 합니다.")
    assert "지금은 '시작 프레임'만 이어져 있습니다" in msg


def test_smooth_with_images_too_is_an_error():
    msg = logic.check_inputs("smooth", {"video": "v.mp4", "start": "a.png", "end": "b.png"})
    assert "'늘릴 영상'만 씁니다" in msg
    assert "'시작 프레임' · '끝 프레임'도 이어져 있습니다" in msg


def test_good_inputs_pass_and_prompt_is_ignored_elsewhere():
    assert logic.check_inputs("tween", {"start": "a", "end": "b", "prompt": "x"}) is None
    assert logic.check_inputs("inbetween", {"start": "a", "end": "b", "prompt": "x"}) is None
    assert logic.check_inputs("smooth", {"video": "v", "prompt": "x"}) is None


def test_messages_fit_the_app_error_limit():
    cases = [("tween", {}), ("inbetween", {"end": 1}), ("smooth", {"start": 1, "end": 1}), ("smooth", {"video": 1, "start": 1})]
    for mode, inputs in cases:
        assert len(logic.check_inputs(mode, inputs)) <= 300


# ── 크기 맞추기 ──────────────────────────────────────────────────────────


def _img(w, h, color=(200, 30, 30)):
    return Image.new("RGB", (w, h), color)


@pytest.mark.parametrize("w,h", [(1024, 640), (640, 1024), (4000, 3000), (3000, 4000), (300, 200), (4000, 400)])
@pytest.mark.parametrize("fit", ["pad", "crop"])
def test_fit_frame_always_gives_the_model_size(w, h, fit):
    canvas, box = logic.fit_frame(_img(w, h), fit, (512, 320))
    assert canvas.size == (512, 320)
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 320


def test_pad_keeps_the_whole_picture_and_box_matches_its_ratio():
    canvas, box = logic.fit_frame(_img(640, 1024), "pad", (512, 320))
    x0, y0, x1, y1 = box
    assert (y0, y1) == (0, 320)
    assert abs((x1 - x0) / (y1 - y0) - 640 / 1024) < 0.02
    assert canvas.getpixel((256, 160)) == (200, 30, 30)


def test_crop_fills_the_canvas():
    canvas, box = logic.fit_frame(_img(640, 1024), "crop", (512, 320))
    assert box == (0, 0, 512, 320)


def test_16_10_picture_has_no_padding():
    _, box = logic.fit_frame(_img(1024, 640), "pad", (512, 320))
    assert box == (0, 0, 512, 320)


def test_match_end_to_start_size():
    start, end = _img(1024, 640), _img(800, 800, (0, 0, 255))
    out = logic.match_size(end, start.size)
    assert out.size == (1024, 640)
    assert logic.match_size(start, start.size) is start


@pytest.mark.parametrize("w,h,expect", [(1024, 640, (960, 600)), (640, 1024, (600, 960)), (512, 320, (512, 320)), (1921, 1081, (960, 540)), (99, 51, (98, 50))])
def test_animeinterp_size_caps_long_side_and_is_even(w, h, expect):
    assert logic.animeinterp_size(w, h) == expect


# ── 반복 프레임 · 채울 시각 ──────────────────────────────────────────────


def _frames(pattern, size=(64, 40)):
    """pattern 의 글자마다 한 프레임. 같은 글자 = 같은 그림(작은 잡음을 더해 압축 흔적 흉내)."""
    rng = np.random.default_rng(0)
    base = {}
    out = []
    for ch in pattern:
        if ch not in base:
            base[ch] = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        noise = rng.integers(-1, 2, base[ch].shape)
        out.append(np.clip(base[ch].astype(int) + noise, 0, 255).astype(np.uint8))
    return out


def test_dedupe_finds_held_drawings():
    keys = logic.dedupe_timeline(_frames("AABBBCD"), logic.DEDUPE_THRESHOLD, 24.0)
    assert [i for i, _ in keys] == [0, 2, 5, 6]
    assert keys[1][1] == pytest.approx(2 / 24)


def test_small_change_is_a_new_drawing():
    """화면 한쪽의 작은 변화(캐릭터 손 하나)도 다른 그림이다 — 화면 전체 평균으로 보면 놓친다."""
    a = np.full((320, 512, 3), 200, np.uint8)
    b = a.copy()
    b[100:130, 200:230] = 30  # 30×30 점이 어두워짐(화면의 0.5%)
    keys = logic.dedupe_timeline([a, a, b, b], logic.DEDUPE_THRESHOLD, 24.0)
    assert [i for i, _ in keys] == [0, 2]


def test_still_video_has_one_key():
    assert [i for i, _ in logic.dedupe_timeline(_frames("AAAAAA"), logic.DEDUPE_THRESHOLD, 24.0)] == [0]


def test_fill_times_keeps_length_for_still_and_short_videos():
    for pattern in ("AAAAAA", "A", "AB", "ABCDEFGHIJKL"):  # 정지 화면 · 1프레임 · 1초 미만
        frames = _frames(pattern)
        keys = [i for i, _ in logic.dedupe_timeline(frames, logic.DEDUPE_THRESHOLD, 24.0)]
        for factor in (2, 3):
            plan = logic.fill_times(keys, len(frames), factor)
            assert len(plan) == len(frames) * factor


def test_fill_times_still_video_copies_the_only_drawing():
    plan = logic.fill_times([0], 6, 2)
    assert plan == [(0, 0, 0.0)] * 12


def test_fill_times_skips_holds_and_spreads_evenly():
    # A A B B → 원래 그림 시각 0, 2. ×2 면 출력 8장: A 시각 0, 그 뒤 B 시각 4(출력 단위)까지 고르게.
    plan = logic.fill_times([0, 2], 4, 2)
    assert plan[0] == (0, 0, 0.0)
    assert plan[1] == (0, 2, pytest.approx(0.25))
    assert plan[2] == (0, 2, pytest.approx(0.5))
    assert plan[3] == (0, 2, pytest.approx(0.75))
    assert plan[4] == (2, 2, 0.0)
    assert plan[7] == (2, 2, 0.0)  # 마지막 그림 뒤에는 다음 그림이 없어 그대로


def test_fill_times_keep_mode_uses_every_frame():
    plan = logic.fill_times(list(range(3)), 3, 3)
    assert plan[:4] == [(0, 0, 0.0), (0, 1, pytest.approx(1 / 3)), (0, 1, pytest.approx(2 / 3)), (1, 1, 0.0)]


def test_pairs_to_compute_are_unique_and_skip_copies():
    plan = logic.fill_times([0, 2], 4, 2)
    assert logic.pairs_of(plan) == {(0, 2): [0.25, 0.5, 0.75]}


# ── 길이 · 쉬운 오류 문구 ───────────────────────────────────────────────


def test_length_message():
    assert logic.length_problem(10.0, 60) is None
    assert logic.length_problem(75.4, 60) == "60초까지 받습니다. 지금 영상은 76초입니다."


def test_out_of_memory_message_is_plain_korean():
    msg = logic.friendly_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "tween", 12282)
    assert msg == (
        "그래픽카드 메모리가 모자랍니다(이 모드 약 10 GB 필요, 이 PC 12 GB). "
        "다른 프로그램을 닫고 다시 하거나 '두 프레임 → 중간 프레임'을 쓰세요."
    )
    assert "Tried" not in msg


def test_out_of_memory_by_type_name_and_cpu_only():
    class OutOfMemoryError(RuntimeError):
        pass

    assert "그래픽카드 메모리가 모자랍니다" in logic.friendly_error(OutOfMemoryError("x"), "smooth", 8192)
    assert logic.friendly_error(ValueError("다른 오류"), "tween", 12282) is None


def test_cpu_only_message_for_tween():
    msg = logic.device_problem("tween", "cpu")
    assert msg.startswith("'두 프레임 → 영상'은 NVIDIA 그래픽카드가 있어야 합니다.")
    assert logic.device_problem("tween", "cuda") is None
    assert logic.device_problem("inbetween", "cpu") is None
    assert logic.device_problem("smooth", "cpu") is None
