"""앱 화면 캡처(app_scene.js 가 찍은 1600×900)를 README 용으로 잘라 docs/ko · docs/en 에 둔다.

  node docs/scripts/app_scene.js <보드 주소> "<워크플로우 이름>" <카드 a> <카드 b> <접두어> <first|both> <ko|en>
    (PromBoard 저장소에서 — node_modules/playwright 를 쓴다. 캔버스를 비우고 워크플로우를 놓고 예제 카드를 끌어 놓은 뒤 실행해 찍는다)
  python docs/scripts/crop_app_shots.py <캡처 폴더>
"""
import os
import sys

from PIL import Image

DOCS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (캡처 파일, 저장 이름, 자를 상자) — 상자는 그 캡처의 노드 자리에 맞춘 값
CROPS = [
    ("tween-selected.png", "ko/app-tween.png", (0, 60, 1060, 620)),
    ("inb-selected.png", "ko/app-inbetween.png", (0, 60, 1060, 600)),
    ("tween-selected2.png", "ko/app-smooth.png", (720, 200, 1500, 590)),
    ("en-tween-selected.png", "en/app-tween.png", (0, 60, 1170, 680)),
    ("en-inb-selected.png", "en/app-inbetween.png", (0, 60, 1170, 680)),
    ("en-tween-selected2.png", "en/app-smooth.png", (680, 200, 1580, 640)),
]

for src, dst, box in CROPS:
    im = Image.open(os.path.join(sys.argv[1], src)).convert("RGB").crop(box)
    out = os.path.join(DOCS, dst)
    im.save(out, optimize=True)
    print(dst, im.size, os.path.getsize(out) // 1024, "KB")
