"""Compact plotting helpers on Pillow (matplotlib is unavailable offline).

Renders at 2x supersampling and downsamples, so lines/labels stay crisp.
Provides: line plot with markers, grouped bars, scatter, legend, grid.
"""

import math

from PIL import Image, ImageDraw, ImageFont

SS = 2  # supersample factor
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]
FONT_BOLD = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\arialbd.ttf"]

PALETTE = [
    (31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


def _font(size, bold=False):
    for path in (FONT_BOLD if bold else FONT_CANDIDATES) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def nice_ticks(lo, hi, target=7):
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 10))
        v += step
    return ticks


def fmt_num(v, step):
    if step >= 1 and abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    dec = max(0, min(4, -int(math.floor(math.log10(step))) + 1))
    return f"{v:.{dec}f}"


class Canvas:
    def __init__(self, width=1400, height=850, title="", xlabel="", ylabel="",
                 margins=(120, 40, 80, 120), bg="white"):
        self.w, self.h = width, height
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.l, self.r, self.t, self.b = margins
        self.img = Image.new("RGB", (width * SS, height * SS), bg)
        self.d = ImageDraw.Draw(self.img)
        self.xlim = None
        self.ylim = None
        self.xticks = None
        self.yticks = None

    def set_scale(self, xlim=None, ylim=None, xticks=None, yticks=None):
        self.xlim, self.ylim = xlim, ylim
        self.xticks, self.yticks = xticks, yticks

    def _px(self, x):
        x0, x1 = self.xlim
        return (self.l + (x - x0) / (x1 - x0) * (self.w - self.l - self.r)) * SS

    def _py(self, y):
        y0, y1 = self.ylim
        return (self.h - self.b - (y - y0) / (y1 - y0) * (self.h - self.t - self.b)) * SS

    def frame(self):
        d = self.d
        f_tick = _font(13 * SS)
        f_lbl = _font(15 * SS)
        f_title = _font(19 * SS, bold=True)
        yt = self.yticks if self.yticks is not None else nice_ticks(*self.ylim)
        xt = self.xticks if self.xticks is not None else nice_ticks(*self.xlim)
        yt = [t for t in yt if self.ylim[0] - 1e-9 <= t <= self.ylim[1] + 1e-9]
        xt = [t for t in xt if self.xlim[0] - 1e-9 <= t <= self.xlim[1] + 1e-9]
        ystep = (yt[1] - yt[0]) if len(yt) > 1 else 1
        xstep = (xt[1] - xt[0]) if len(xt) > 1 else 1
        for y in yt:
            py = self._py(y)
            d.line([(self.l * SS, py), ((self.w - self.r) * SS, py)], fill=(228, 230, 234), width=1)
            d.text(((self.l - 12) * SS, py), fmt_num(y, ystep), font=f_tick,
                   fill=(70, 74, 80), anchor="rm")
        for x in xt:
            px = self._px(x)
            d.line([(px, (self.h - self.b) * SS), (px, (self.h - self.b) * SS + 7 * SS)],
                   fill=(90, 94, 100), width=1)
            d.text((px, (self.h - self.b + 12) * SS), fmt_num(x, xstep), font=f_tick,
                   fill=(70, 74, 80), anchor="mm")
        d.line([(self.l * SS, self.t * SS - 6 * SS), (self.l * SS, (self.h - self.b) * SS)],
               fill=(60, 64, 70), width=2)
        d.line([(self.l * SS, (self.h - self.b) * SS), ((self.w - self.r) * SS, (self.h - self.b) * SS)],
               fill=(60, 64, 70), width=2)
        if self.title:
            d.text((self.l * SS, 20 * SS), self.title, font=f_title, fill=(28, 32, 40))
        if self.xlabel:
            d.text((((self.l + self.w - self.r) / 2) * SS, (self.h - 52) * SS), self.xlabel,
                   font=f_lbl, fill=(50, 54, 60), anchor="mm")
        if self.ylabel:
            tmp = Image.new("RGBA", (200 * SS, 26 * SS), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).text((100 * SS, 13 * SS), self.ylabel, font=f_lbl,
                                     fill=(50, 54, 60), anchor="mm")
            tmp = tmp.rotate(90, expand=True)
            self.img.paste(tmp, (int(18 * SS), int((self.t + (self.h - self.t - self.b) / 2 - tmp.height / SS / 2) * SS)), tmp)

    def line(self, xs, ys, color_idx=0, label=None, width=3, marker=5, dash=False, color=None):
        d = self.d
        color = color or PALETTE[color_idx % len(PALETTE)]
        pts = [(self._px(x), self._py(y)) for x, y in zip(xs, ys)]
        for a, b in zip(pts, pts[1:]):
            if dash:
                self._dashed(a, b, color, width * SS)
            else:
                d.line([a, b], fill=color, width=width * SS)
        if marker:
            r = marker * SS / 2
            for cx, cy in pts:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                          outline=(255, 255, 255), width=2)
        return color

    def _dashed(self, a, b, color, width, seg=9 * SS, gap=7 * SS):
        x1, y1 = a
        x2, y2 = b
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return
        ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
        t = 0.0
        while t < dist:
            e = min(t + seg, dist)
            self.d.line([(x1 + ux * t, y1 + uy * t), (x1 + ux * e, y1 + uy * e)],
                        fill=color, width=width)
            t = e + gap

    def bars(self, groups, series, labels, colors=None, xticklabels=None, bar_gap=0.14):
        """groups: list of x centers (int index); series: list of (name, values)."""
        d = self.d
        n = len(series)
        span = 1.0 / (n + (n + 1) * bar_gap)
        # baseline drawn by frame
        for si, (name, values) in enumerate(series):
            color = (colors or PALETTE)[si % len(PALETTE)]
            bw = span * 2.0
            for gi, v in enumerate(values):
                cx = self._px(groups[gi])
                x0 = cx + (si - (n - 1) / 2) * (bw * SS)
                half = bw * SS / 2
                y0 = self._py(max(v, 0))
                yb = self._py(0)
                d.rectangle([x0 - half, min(y0, yb), x0 + half, max(y0, yb)], fill=color)
        if xticklabels:
            f = _font(12 * SS)
            for gi, lab in enumerate(xticklabels):
                d.text((self._px(groups[gi]), (self.h - self.b + 12) * SS), lab,
                       font=f, fill=(70, 74, 80), anchor="mm")

    def hbar(self, labels, values, colors=None, vmax=None, fmt="{:.0f}", bar_h=0.55):
        d = self.d
        y0, y1 = self.ylim
        n = len(labels)
        for i, (lab, v) in enumerate(zip(labels, values)):
            y = y1 - (i + 0.5) * (y1 - y0) / n
            color = (colors or PALETTE)[i % len(PALETTE)]
            xa = self._px(self.xlim[0])
            xb = self._px(v)
            py = self._py(y)
            half = (self._py(y1 - (y1 - y0) / n * (1 - bar_h) / 2) - self._py(y1))
            d.rectangle([xa, py - abs(half), xb, py + abs(half)], fill=color)
            d.text((xa - 10 * SS, py), lab, font=_font(13 * SS), fill=(50, 54, 60), anchor="rm")
            d.text((xb + 8 * SS, py), fmt.format(v), font=_font(12 * SS),
                   fill=(60, 64, 70), anchor="lm")

    def scatter(self, xs, ys, color_idx=0, label=None, r=6, alpha_blend=False):
        d = self.d
        color = PALETTE[color_idx % len(PALETTE)]
        for x, y in zip(xs, ys):
            cx, cy = self._px(x), self._py(y)
            rr = r * SS / 2
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color, outline=(255, 255, 255), width=2)

    def legend(self, entries, pos="tr", title=None):
        """entries: list of (label, color, style) where style in {'line','bar','dot'}."""
        d = self.d
        f = _font(13 * SS)
        fb = _font(13 * SS, bold=True)
        pad = 12 * SS
        row_h = 24 * SS
        widths = []
        for label, _, _ in entries:
            widths.append(d.textlength(label, font=f))
        box_w = max(widths) + 70 * SS + 2 * pad
        box_h = row_h * len(entries) + 2 * pad + (26 * SS if title else 0)
        if pos == "tr":
            x1 = (self.w - self.r - 14) * SS
            y0 = (self.t + 8) * SS
        elif pos == "tl":
            x1 = (self.l + 300) * SS
            y0 = (self.t + 8) * SS
        else:  # bottom left
            x1 = (self.l + 300) * SS
            y0 = (self.h - self.b - 20) * SS - box_h
        x0 = x1 - box_w
        d.rectangle([x0, y0, x1, y0 + box_h], fill=(252, 252, 253), outline=(214, 216, 220), width=2)
        yy = y0 + pad
        if title:
            d.text((x0 + pad, yy), title, font=fb, fill=(45, 49, 55))
            yy += 26 * SS
        for i, (label, color, style) in enumerate(entries):
            cy = yy + row_h / 2
            if style == "line":
                d.line([(x0 + pad, cy), (x0 + pad + 30 * SS, cy)], fill=color, width=3 * SS)
                r = 3 * SS
                d.ellipse([x0 + pad + 15 * SS - r, cy - r, x0 + pad + 15 * SS + r, cy + r], fill=color)
            elif style == "bar":
                d.rectangle([x0 + pad, cy - 7 * SS, x0 + pad + 30 * SS, cy + 7 * SS], fill=color)
            else:
                r = 5 * SS
                d.ellipse([x0 + pad + 15 * SS - r, cy - r, x0 + pad + 15 * SS + r, cy + r], fill=color)
            d.text((x0 + pad + 42 * SS, cy), label, font=f, fill=(45, 49, 55), anchor="lm")
            yy += row_h

    def annotate(self, x, y, text, dx=8, dy=-14, color=(60, 64, 70), size=12):
        self.d.text((self._px(x) + dx * SS, self._py(y) + dy * SS), text,
                    font=_font(size * SS), fill=color, anchor="lm")

    def save(self, path, quality=95):
        out = self.img.resize((self.w, self.h), Image.LANCZOS)
        out.save(path, quality=quality)
        return path
