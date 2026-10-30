# NOTICE — 담아 온 코드 · 받는 모델 · 예제 그림

이 플러그인(`promboard.anime2d`)은 아래 공개 코드를 `nodes/` 아래에 최소로 고쳐 담았다. 고친 줄에는 `PromBoard 고침` 주석을 달았다.

## 담아 온 코드

| 폴더 | 원본 | 커밋 | 라이선스 |
|---|---|---|---|
| `nodes/tooncrafter/` | [Doubiiu/ToonCrafter](https://github.com/Doubiiu/ToonCrafter) — `lvdm/` · `utils/utils.py` · `configs/inference_512_v1.0.yaml` 일부 | `b0c47ff339c5e5ec45b84d0c6587850f242d41ef` | Apache-2.0 (`nodes/tooncrafter/LICENSE`, Copyright Tencent) |
| `nodes/animeinterp/models/` | [lisiyao21/AnimeInterp](https://github.com/lisiyao21/AnimeInterp) — `models/` 의 `AnimeInterp_no_cupy.py` · `GridNet.py` · `forward_warp2.py` · `rfr_model/` | `223c5e6f39c525bdfcb51ca8acf9d59d4adeb643` | MIT (원본 README 의 "Our code is released under MIT License." — 원본 저장소에 LICENSE 파일은 없다) |
| `nodes/animeinterp/models/rfr_model/` | 위 AnimeInterp 안의 RAFT 기반 코드 ([princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)) | 같음 | BSD-3-Clause (`nodes/animeinterp/models/rfr_model/LICENSE`) |

`nodes/main.py` · `nodes/logic.py` · `tests/` 는 PromBoard 가 쓴 코드다(Apache-2.0, 저장소 `LICENSE`).

### 고친 곳

- ToonCrafter
  - `lvdm/modules/attention.py` · `lvdm/models/autoencoder_dualref.py`: xformers `memory_efficient_attention` → torch `scaled_dot_product_attention`(xformers 없이 돈다). `(B*H, N, D)` 를 `(B, H, N, D)` 로 풀어 넣는다.
  - `lvdm/modules/encoders/condition.py`: 쓰지 않는 T5 · OpenAI CLIP 글 인코더의 `transformers` 를 그 클래스 안에서 불러온다(transformers 를 받지 않게).
  - `utils/utils.py`: 쓰지 않는 `resize_numpy_image` 의 `cv2` 를 그 함수 안에서 불러온다(opencv 를 받지 않게).
  - `lvdm/models/autoencoder_dualref.py` · `lvdm/modules/attention_svd.py`: torch 2.0 미만에서만 나는 오류 문구의 설치 명령 글자를 뺐다(PromBoard 레지스트리 검사가 실행 중 설치로 본다).
  - 고정 버전(torch 2.0 · pytorch_lightning 1.9.3 · transformers 4.25)은 풀었다 — 잠금 파일 `nodes/requirements-*.txt`(torch 2.8.0 · pytorch-lightning 2.6 · open-clip-torch 2.22.0).
  - 코드 밖에서 바꾼 것(`nodes/main.py`): open_clip `pretrained=None`(ViT-H-14 가중치는 `model.ckpt` 안에 있다), 모델 fp16, `torch.load(weights_only=True, mmap=True)`, 장치는 PromBoard 가 정한 `ctx.device`.
- AnimeInterp
  - `models/__init__.py`: cupy 가 필요한 `AnimeInterp` 를 빼고 `AnimeInterpNoCupy` 만 내보낸다.
  - `models/rfr_model/rfr_new.py` · `utils.py`: `.cuda()` → 입력과 같은 장치(CPU 도 돈다), `grid_sample(align_corners=True)`(원본 README 의 torch 1.3 이상 안내), 쓰지 않는 함수의 `scipy` 를 그 함수 안에서 불러온다.
  - SGM(Segment-Guided Matching) 첫 흐름은 쓰지 않는다 — 흐름을 0 에서 시작한다(softsplat · torch_scatter 를 받지 않으려고).

## 받는 모델 (`manifest.json` `requires.models`)

| 모델 | 받는 곳 | 크기 | 라이선스 |
|---|---|---|---|
| ToonCrafter 512 보간판 `model.ckpt` | [Doubiiu/ToonCrafter](https://huggingface.co/Doubiiu/ToonCrafter) (커밋 `7c56c5a23d9f8a9d99398e2a2491fff4bd6cffaf`) | 10.5GB | Apache-2.0 (그 저장소 LICENSE · 모델 카드 "Feel free to use it under the Apache-2.0 license") |
| AnimeInterp `anime_interp_full.ckpt` | 원 저자 Dropbox(README 링크) 파일을 그대로 [이 저장소 릴리스 `weights-animeinterp`](https://github.com/promboard-app/anime2d/releases/tag/weights-animeinterp)에 다시 올림 | 24.9MB | 적혀 있지 않음(`LicenseRef-unspecified`). 원 저자 README 의 MIT 는 코드 몫이다 |

AnimeInterp 가중치 출처: Li Siyao, Shiyu Zhao, Weijiang Yu, Wenxiu Sun, Dimitris Metaxas, Chen Change Loy, Ziwei Liu, "Deep Animation Video Interpolation in the Wild", CVPR 2021. 원 파일 sha256 `0e7efcadf59b4b02e2a7122662bade0be2d8d057302eb772b602aaa01090afd1`.

## 예제 그림 (`cards/`)

`cards/` 의 그림 6장은 PromBoard 가 AI(OpenAI codex CLI 의 이미지 생성)로 새로 그렸다. 공개 데이터셋 · 저장소 예제 그림은 재배포가 안 되어 넣지 않았다 — 그림체를 참고하는 데만 이 PC 에서 썼고, 캐릭터 · 로고 · 글자 · 구도는 따라 그리지 않았다.
