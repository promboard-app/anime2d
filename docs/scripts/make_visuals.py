"""README 그림 · 움짤 만들기 — make_results.py 가 만든 결과 폴더에서 docs/*.gif 를 만든다.

  python docs/scripts/make_visuals.py <결과 폴더>

결과 폴더 안: tween/ (컬러 두 프레임 → 영상, tween-00~15.png) · line/ (선화 두 프레임 → 영상) ·
inbetween/ (between-00~06.png) · smooth/ (smooth.mp4, 8fps 영상을 ×3) · smooth-src/ (늘리기 전 8fps 영상 tween.mp4).
워크플로우 썸네일 workflows/*.jpg 도 같은 결과로 만든다. 움짤에는 글자를 넣지 않는다(한 · 영 README 가 같이 쓴다). 설명은 README 문장으로 단다.
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
CARDS = os.path.join(os.path.dirname(DOCS), "cards")
GAP = 8
BG = (255, 255, 255)


def fit(img, h):
    return img.convert("RGB").resize((round(img.width * h / img.height), h), Image.LANCZOS)


def row(images, h):
    tiles = [fit(im, h) for im in images]
    w = sum(t.width for t in tiles) + GAP * (len(tiles) - 1)
    out = Image.new("RGB", (w, h), BG)
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + GAP
    return out


def save_gif(frames, path, ms, hold_last=0):
    """frames: PIL 목록. ms: 한 장 시간. 끝에서 hold_last 장만큼 멈췄다가 처음으로."""
    durations = [ms] * len(frames)
    if hold_last:
        durations[-1] += hold_last
        durations[0] += hold_last
    pal = frames[len(frames) // 2].quantize(colors=255, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(path, os.path.getsize(path) // 1024, "KB", frames[0].size, len(frames), "frames")


def read_video(path):
    import av

    with av.open(path) as c:
        return [Image.fromarray(f.to_ndarray(format="rgb24")) for f in c.decode(c.streams.video[0])]


def side_by_side_anim(start, frames, end, h, path, ms, hold):
    """시작 프레임 | 결과 움직임 | 끝 프레임."""
    a, b = fit(start, h), fit(end, h)
    out = []
    for f in frames:
        out.append(row([a, f, b], h))
    save_gif(out, path, ms, hold)


def main():
    res = sys.argv[1]
    def load(d, n):
        names = sorted(x for x in os.listdir(os.path.join(res, d)) if x.startswith(n + "-") and x.endswith(".png"))
        return [Image.open(os.path.join(res, d, x)) for x in names]

    tween = load("tween", "tween")
    side_by_side_anim(Image.open(os.path.join(CARDS, "color-start.jpg")), tween, Image.open(os.path.join(CARDS, "color-end.jpg")),
                      200, os.path.join(DOCS, "hero.gif"), 125, 700)
    line = load("line", "tween")
    side_by_side_anim(Image.open(os.path.join(CARDS, "line-start.png")), line, Image.open(os.path.join(CARDS, "line-end.png")),
                      200, os.path.join(DOCS, "tween-line.gif"), 125, 700)

    # 중간 프레임: 얼굴 · 손 부분을 크게(움직임이 작아서), 시작 → 중간 7장 → 끝 → 거꾸로
    box = (300, 30, 700, 330)  # 960×600 기준
    start = Image.open(os.path.join(CARDS, "cont-start.jpg")).resize((960, 600), Image.LANCZOS)
    end = Image.open(os.path.join(CARDS, "cont-end.jpg")).resize((960, 600), Image.LANCZOS)
    mids = load("inbetween", "between")
    seq = [start] + mids + [end]
    seq = seq + seq[-2:0:-1]
    crops = [im.convert("RGB").crop(box) for im in seq]
    full = [row([s.crop(box), c, end.crop(box)], 260) for s, c in zip([start] * len(crops), crops)]
    save_gif(full, os.path.join(DOCS, "inbetween.gif"), 83, 0)

    # 워크플로우 썸네일(설치 zip 에 든다 — workflows/): 실제 결과 프레임 한 줄
    wf = os.path.join(os.path.dirname(DOCS), "workflows")
    # 앱 워크플로우 칸은 정사각으로 잘라 보인다 — 처음 · 끝 프레임을 위아래로
    a, b = fit(tween[0], 250), fit(tween[-1], 250)
    thumb = Image.new("RGB", (a.width, 500 + GAP), BG)
    thumb.paste(a, (0, 0))
    thumb.paste(b, (0, 250 + GAP))
    thumb.save(os.path.join(wf, "two-frames-to-video-smooth.jpg"), quality=86)
    thumb = row([start, mids[len(mids) // 2], end], 240)
    thumb.save(os.path.join(wf, "two-frames-inbetween.jpg"), quality=86)
    print("thumbnails", os.path.getsize(os.path.join(wf, "two-frames-to-video-smooth.jpg")) // 1024, "KB",
          os.path.getsize(os.path.join(wf, "two-frames-inbetween.jpg")) // 1024, "KB")

    # 프레임 늘리기: 왼쪽 원래 8fps · 오른쪽 ×3 = 24fps. 같은 2초를 24fps 시계로 맞춰 나란히.
    src = read_video(os.path.join(res, "smooth-src", "tween.mp4"))
    smooth = read_video(os.path.join(res, "smooth", "smooth.mp4"))
    pairs = [row([src[min(len(src) - 1, i // 3)], f], 220) for i, f in enumerate(smooth)]
    save_gif(pairs, os.path.join(DOCS, "smooth.gif"), 42, 500)


if __name__ == "__main__":
    main()
