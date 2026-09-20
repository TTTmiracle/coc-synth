"""Measure the game's own geometry off real screenshots.

Everything here is measurement, not modelling: the tile lattice period, and from
it the scale the game draws each building at. The published icons carry no
inter-building scale -- each is framed to fill its own canvas -- so the only way
to know that a 2x2 Hidden Tesla is short and a 4x4 Town Hall is tall is to look
at a picture of the game and measure.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def highpass(a: np.ndarray, k: int = 9) -> np.ndarray:
    """Strip slow shading so the correlation sees texture, not lighting."""
    ker = np.ones(k) / k
    sm = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 1, a)
    sm = np.apply_along_axis(lambda c: np.convolve(c, ker, "same"), 0, sm)
    return a - sm


def _ncc_shift(p: np.ndarray, dx: int, dy: int) -> float:
    h, w = p.shape
    A = p[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
    B = p[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    n, m = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    A, B = A[:n, :m], B[:n, :m]
    denom = np.sqrt((A * A).sum() * (B * B).sum())
    return float((A * B).sum() / denom) if denom else 0.0


def _refine(vals: list[float], lo: int) -> tuple[float, float]:
    i = int(np.argmax(vals))
    if 0 < i < len(vals) - 1:
        a, b, c = vals[i - 1], vals[i], vals[i + 1]
        d = (a - c) / (2 * (a - 2 * b + c)) if (a - 2 * b + c) else 0.0
        return lo + i + d, vals[i]
    return float(lo + i), vals[i]


def tile_period(path: str, patches: list[tuple[int, int]] | None = None
                ) -> tuple[float, float, float]:
    """Tile width and height in pixels, from the ground's checkerboard.

    Correlating the high-passed image against shifted copies of itself finds the
    lattice period directly, and a parabola through the peak gives it to well
    under a pixel -- far finer than an FFT bin over a patch this size.
    """
    g = np.asarray(Image.open(path).convert("L")).astype(float)
    H, W = g.shape
    patches = patches or [(W // 2, H // 2)]
    xs, ys = [], []
    for cx, cy in patches:
        ph, pw = 560, 960
        y0, x0 = max(0, cy - ph // 2), max(0, cx - pw // 2)
        p = highpass(g[y0:y0 + ph, x0:x0 + pw].astype(float))
        p -= p.mean()
        px, cxv = _refine([_ncc_shift(p, d, 0) for d in range(24, 46)], 24)
        py, cyv = _refine([_ncc_shift(p, 0, d) for d in range(16, 36)], 16)
        if cxv > 0.15 and cyv > 0.15:
            xs.append(px)
            ys.append(py)
    if not xs:
        return (0.0, 0.0, 0.0)
    tw, th = float(np.median(xs)), float(np.median(ys))
    return tw, th, tw / th


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        tw, th, _ = tile_period(path, [(700, 600), (1200, 540), (1700, 600),
                                       (1200, 300), (1200, 820)])
        if not tw:
            print(f"{path:28s} no lattice found (ground too covered)")
            continue
        print(f"{path:28s} tile_w={tw:6.2f}  tile_h={th:6.2f}  aspect={th / tw:.4f}")


def _corr(A: np.ndarray, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Cross-correlate A with kernel K over the whole image, via FFT."""
    FA = np.fft.rfft2(A, s=shape)
    FK = np.fft.rfft2(K, s=shape)
    return np.fft.irfft2(FA * np.conj(FK), s=shape)


def masked_ncc(image: np.ndarray, tmpl: np.ndarray, mask: np.ndarray) -> float:
    """Best masked normalised cross-correlation of `tmpl` anywhere in `image`.

    Masked because a sprite is mostly transparent: correlating its bounding box
    would score the grass around it as much as the building.
    """
    H, W = image.shape
    th, tw = tmpl.shape
    if th >= H or tw >= W:
        return -1.0
    t = np.where(mask, tmpl, 0.0)
    n = mask.sum()
    if n < 40:
        return -1.0
    t_sum = t.sum()
    t_sq = (t * t).sum()
    t_var = t_sq - t_sum * t_sum / n
    if t_var <= 0:
        return -1.0

    s1 = _corr(image, mask.astype(float), (H, W))
    s2 = _corr(image * image, mask.astype(float), (H, W))
    s3 = _corr(image, t, (H, W))
    num = s3 - s1 * (t_sum / n)
    i_var = s2 - s1 * s1 / n
    den = np.sqrt(np.maximum(i_var, 1e-9) * t_var)
    ncc = num / den
    ncc[H - th + 1:, :] = -1
    ncc[:, W - tw + 1:] = -1
    return float(np.nanmax(ncc))


def best_scale(screenshot: str, sprite_path: str, tile_w: float, footprint: int = 3,
               lo_frac: float = 0.35, hi_frac: float = 1.35, steps: int = 20
               ) -> tuple[float, float, float]:
    """Which on-screen width, in tiles, this sprite matches the screenshot at.

    Sweeping the scale and keeping the best correlation asks the picture directly
    how big the game draws this building, instead of assuming its icon's framing
    says anything about that.
    """
    rgb = np.asarray(Image.open(screenshot).convert("RGB")).astype(float)
    chans = [highpass(rgb[..., i]) for i in range(3)]
    src = Image.open(sprite_path).convert("RGBA")
    src = src.crop(src.getbbox())
    diamond = footprint * tile_w          # on-screen width of the footprint
    results = []
    for k in range(steps):
        frac = lo_frac + (hi_frac - lo_frac) * k / (steps - 1)
        tiles = frac * footprint
        w = int(round(frac * diamond))
        h = max(1, round(src.height * w / src.width))
        if w < 8 or h < 8:
            continue
        r = src.resize((w, h), Image.Resampling.LANCZOS)
        arr = np.asarray(r.convert("RGB")).astype(float)
        m = np.asarray(r.getchannel("A")) > 128
        # Colour, not luminance: Clash art is strongly coloured and a purple
        # storage against green grass is unmistakable in a way grey levels are not.
        score = float(np.mean([masked_ncc(chans[i], highpass(arr[..., i]), m)
                               for i in range(3)]))
        results.append((score, tiles, h / tile_w))
    if len(results) < 3:
        return (-1.0, 0.0, 0.0)
    # An interior peak only. Normalised correlation rises without bound as the
    # template shrinks towards noise, so a best score sitting on the smallest
    # scale tried is that artefact rather than a match.
    interior = [(results[i][0], results[i][1], results[i][2])
                for i in range(1, len(results) - 1)
                if results[i][0] >= results[i - 1][0] and results[i][0] >= results[i + 1][0]]
    if not interior:
        return (-1.0, 0.0, 0.0)
    return max(interior)


def diamond_fit(path: str) -> dict:
    """Fit the buildable diamond's four edges and report the tile size.

    More reliable than the lattice period on a built-up base: the checkerboard is
    mostly covered by buildings, but the boundary between the bright buildable
    ground and the dark forest around it survives.
    """
    c = np.asarray(Image.open(path).convert("RGB")).astype(int)
    H, W, _ = c.shape
    bright = (c[..., 1] > 150) & (c[..., 0] > 95) & (c[..., 1] - c[..., 2] > 85)
    rows = []
    for y in range(20, H - 10, 2):
        idx = np.nonzero(bright[y])[0]
        if len(idx) < 150:
            continue
        segs = np.split(idx, np.nonzero(np.diff(idx) > 24)[0] + 1)
        s = max(segs, key=len)
        if len(s) > 260:
            rows.append((y, s[0], s[-1]))
    if len(rows) < 40:
        return {}
    a = np.array(rows, dtype=float)
    y, L, R = a[:, 0], a[:, 1], a[:, 2]
    ymid = y[np.argmax(R - L)]

    def fit(ys, xs):
        keep = np.ones(len(ys), bool)
        for _ in range(6):
            m, b = np.polyfit(ys[keep], xs[keep], 1)
            res = np.abs(xs - (m * ys + b))
            keep = res < max(10, np.percentile(res[keep], 65))
            if keep.sum() < 6:
                break
        return m, b

    up, lo = y < ymid - 25, y > ymid + 25
    if up.sum() < 10 or lo.sum() < 10:
        return {}
    mUL, bUL = fit(y[up], L[up]); mUR, bUR = fit(y[up], R[up])
    mLL, bLL = fit(y[lo], L[lo]); mLR, bLR = fit(y[lo], R[lo])
    yl = (bLL - bUL) / (mUL - mLL); xl = mUL * yl + bUL
    yr = (bLR - bUR) / (mUR - mLR); xr = mUR * yr + bUR
    ytop = (bUR - bUL) / (mUL - mUR)
    ybot = (bLR - bLL) / (mLL - mLR)
    slope = float(np.median([abs(mUL), abs(mUR), abs(mLL), abs(mLR)]))
    return {"tile_w": (xr - xl) / 44.0, "tile_h": (ybot - ytop) / 44.0,
            "aspect_from_edges": 1.0 / slope, "width_px": xr - xl,
            "height_px": ybot - ytop, "centre": ((xl + xr) / 2, (ytop + ybot) / 2)}


if False:
    pass


class Scene:
    """A reference screenshot, pre-transformed so many templates can be tried.

    The image side of every correlation is identical no matter which sprite is
    being matched, so it is computed once here. Without that, sweeping 29 types
    across a dozen scales re-runs the same transforms a few thousand times.
    """

    def __init__(self, path: str, tile_w: float, downscale: float = 0.5):
        img = Image.open(path).convert("RGB")
        self.scale = downscale
        size = (int(img.width * downscale), int(img.height * downscale))
        img = img.resize(size, Image.Resampling.LANCZOS)
        self.tile_w = tile_w * downscale
        self.shape = (img.height, img.width)
        arr = np.asarray(img).astype(float)
        self.chan = [highpass(arr[..., i]) for i in range(3)]
        self.f_chan = [np.fft.rfft2(c, s=self.shape) for c in self.chan]
        self.f_sq = [np.fft.rfft2(c * c, s=self.shape) for c in self.chan]

    def _inv(self, fa, fk):
        return np.fft.irfft2(fa * np.conj(fk), s=self.shape)

    def score(self, tmpl: list[np.ndarray], mask: np.ndarray) -> float:
        """How far the best match stands out from a typical one, at this scale.

        Raw peak correlation cannot be compared across scales: a smaller template
        covers fewer pixels, so it fits noise more easily and its peak rises no
        matter whether the scale is right. Sweeping on peak correlation therefore
        always walks downhill to the smallest size tried -- which is how a 4x4 Town
        Hall came out measured at two thirds of a tile.

        Standardising the peak against the mean and spread of the *same*
        correlation map removes that trend, because the easier fitting lifts every
        position in the map, not just the correct one. What survives is how
        distinctive the match is, which is what actually peaks at the right scale.
        """
        th, tw = mask.shape
        H, W = self.shape
        if th >= H or tw >= W:
            return -1.0
        n = mask.sum()
        if n < 30:
            return -1.0
        f_mask = np.fft.rfft2(mask.astype(float), s=self.shape)
        out = []
        for i in range(3):
            t = np.where(mask, tmpl[i], 0.0)
            t_sum, t_sq = t.sum(), (t * t).sum()
            t_var = t_sq - t_sum * t_sum / n
            if t_var <= 0:
                return -1.0
            s1 = self._inv(self.f_chan[i], f_mask)
            s2 = self._inv(self.f_sq[i], f_mask)
            s3 = self._inv(self.f_chan[i], np.fft.rfft2(t, s=self.shape))
            ncc = (s3 - s1 * (t_sum / n)) / np.sqrt(
                np.maximum(s2 - s1 * s1 / n, 1e-9) * t_var)
            valid = ncc[:H - th + 1, :W - tw + 1]
            valid = valid[np.isfinite(valid)]
            if valid.size < 100:
                return -1.0
            sd = valid.std()
            if sd <= 1e-6:
                return -1.0
            out.append(float((valid.max() - valid.mean()) / sd))
        return float(np.mean(out))


def sweep(scene: Scene, sprite_path: str, footprint: int,
          lo: float = 0.30, hi: float = 1.30, steps: int = 13):
    """Best interior peak over a sweep of art-width fractions."""
    src = Image.open(sprite_path).convert("RGBA")
    src = src.crop(src.getbbox())
    diamond = footprint * scene.tile_w
    res = []
    for k in range(steps):
        frac = lo + (hi - lo) * k / (steps - 1)
        w = int(round(frac * diamond))
        h = max(1, round(src.height * w / src.width))
        if w < 10 or h < 10 or h > scene.shape[0] - 2:
            res.append((-1.0, frac, 0.0))
            continue
        r = src.resize((w, h), Image.Resampling.LANCZOS)
        arr = np.asarray(r.convert("RGB")).astype(float)
        m = np.asarray(r.getchannel("A")) > 128
        res.append((scene.score([highpass(arr[..., i]) for i in range(3)], m),
                    frac, h / scene.tile_w))
    peaks = [res[i] for i in range(1, len(res) - 1)
             if res[i][0] >= res[i - 1][0] and res[i][0] >= res[i + 1][0]]
    return max(peaks) if peaks else (-1.0, 0.0, 0.0)
