"""README 그림의 재료(결과 프레임 · 영상)를 노드 코드로 직접 만든다 — 앱 없이 가짜 ctx 로 nodes/main.py 의 fill 을 부른다.

  python docs/scripts/make_results.py <결과 폴더> <ToonCrafter model.ckpt> <anime_interp_full.ckpt>

플러그인 가상환경(torch 2.8 + nodes/requirements-*.txt)의 파이썬으로 돌린다. NVIDIA 그래픽카드가 있어야 한다('두 프레임 → 영상').
만드는 것: tween/ (컬러, 시드 123) · line/ (선화, 시드 123) · inbetween/ (중간 7장) · smooth-src/tween.mp4 · smooth/ (×3).
그 뒤 make_visuals.py <결과 폴더> 로 움짤을 만든다.
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
CARDS = os.path.join(ROOT, "cards")


class Ctx:
    def __init__(self, out_dir, models):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.models, self.device = out_dir, models, "cuda"
        self.cancelled, self.cache_dir, self.keys, self.t0 = False, None, {}, time.time()

    def out_path(self, name):
        return os.path.join(self.out_dir, name)

    def in_path(self, p):
        return p

    def model_path(self, name):
        return self.models[name]

    def progress(self, ratio, stage="work"):
        pass

    def check_cancel(self):
        pass

    def log(self, msg, level="info"):
        print(f"[{time.time() - self.t0:6.1f}s] {msg}", flush=True)


def main():
    out, tc, ai = sys.argv[1:4]
    models = {"Doubiiu/ToonCrafter/model.ckpt": tc, "animeinterp/anime_interp_full.ckpt": ai}
    from nodes import main as node

    c = lambda name: os.path.join(CARDS, name)
    run = lambda d, inputs, settings: node.fill(Ctx(os.path.join(out, d), models), inputs, settings)
    run("tween", {"start": c("color-start.jpg"), "end": c("color-end.jpg")}, {"mode": "tween", "seed": 123})
    run("line", {"start": c("line-start.png"), "end": c("line-end.png")}, {"mode": "tween", "seed": 123})
    run("inbetween", {"start": c("cont-start.jpg"), "end": c("cont-end.jpg")}, {"mode": "inbetween", "count": 7})
    os.makedirs(os.path.join(out, "smooth-src"), exist_ok=True)
    shutil.copy(os.path.join(out, "tween", "tween.mp4"), os.path.join(out, "smooth-src", "tween.mp4"))
    run("smooth", {"video": os.path.join(out, "smooth-src", "tween.mp4")}, {"mode": "smooth", "factor": "x3"})


if __name__ == "__main__":
    main()
