# PromBoard 고침: 원본 models/__init__.py 는 cupy 가 필요한 AnimeInterp 도 불러온다 — cupy 없는 AnimeInterpNoCupy 만 내보낸다.
from .AnimeInterp_no_cupy import AnimeInterpNoCupy

__all__ = ['AnimeInterpNoCupy']
