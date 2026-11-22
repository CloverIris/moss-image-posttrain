#!/usr/bin/env python3
"""Create a deterministic reward report with untouched full-size image tiles."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    report = json.loads(Path(cfg['report']).read_text(encoding='utf-8-sig'))
    rows = report['ranking']
    assert len(rows) == cfg.get('expected_images', 16)
    assert len({row['image'] for row in rows}) == len(rows)
    assert [row['rank'] for row in rows] == list(range(1, len(rows)+1))
    assert all(rows[i]['final_reward'] >= rows[i+1]['final_reward'] for i in range(len(rows)-1))
    source = Path(cfg['source_dir']).resolve()
    images, manifest = [], []
    for row in rows:
        assert Path(row['image']).name == row['image']
        path = source / row['image']
        assert path.resolve().parent == source
        assert math.isfinite(row['final_reward'])
        assert abs(.5 + sum(row['contributions'].values()) - row['final_reward']) < 1e-6
        with Image.open(path) as original:
            image = original.convert('RGB')
        assert image.size == (1024, 1024)
        images.append(image)
        manifest.append(dict(rank=row['rank'], image=str(path), reward=row['final_reward'],
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    regular = cfg.get('font', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold = cfg.get('bold_font', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    font = lambda size: ImageFont.truetype(regular, size)
    strong = lambda size: ImageFont.truetype(bold, size)
    columns = cfg.get('columns', 4)
    margin, gap, frame, footer, header = 24, 24, 8, 184, 188
    cell_width = 1024 + 2*frame
    cell_height = 1024 + 2*frame + footer
    row_count = math.ceil(len(rows) / columns)
    width = 2*margin + columns*cell_width + (columns-1)*gap
    height = header + row_count*cell_height + (row_count-1)*gap + margin
    sheet = Image.new('RGB', (width, height), '#edf1f7')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 18), 'MOSS ART | same-prompt reward ranking | rollout @ update 6', font=strong(48), fill='#15243d')
    draw.text((margin, 82), '16 original 1024 x 1024 images | descending reward | 0070.png outlined orange', font=font(32), fill='#45536b')
    draw.text((margin, 128), 'Final reward = 0.5 + Kawai + DeepGHS + HPS + SDG; K/D/H below are calibrated weighted contributions.', font=font(29), fill='#45536b')
    for i, (row, image, item) in enumerate(zip(rows, images, manifest)):
        x = margin + (i % columns)*(cell_width+gap)
        y = header + (i // columns)*(cell_height+gap)
        selected = row['image'] == cfg.get('highlight_image', '0070.png')
        draw.rectangle((x, y, x+cell_width-1, y+cell_height-1), fill='#dd8d20' if selected else '#d1dae7')
        draw.rectangle((x+frame, y+1024+2*frame, x+cell_width-frame-1, y+cell_height-frame-1), fill='#fff1d9' if selected else 'white')
        tile = (x+frame, y+frame, x+frame+1024, y+frame+1024)
        sheet.paste(image, tile[:2])
        item['tile_box'] = tile
        label_x, label_y = x+18, y+1024+2*frame+8
        lines = [
            (f"#{row['rank']:02d}  {row['image']}   Reward {row['final_reward']:.6f}", strong(36), '#a85600' if selected else '#15243d'),
            (f"Aesthetic {row['aesthetic_reward']:.6f}    SDG {row['artifact_penalty']:+.6f}", font(31), '#233950'),
            (f"K {row['contributions']['kawai']:+.6f}   D {row['contributions']['deepghs']:+.6f}   H {row['contributions']['hps']:+.6f}", font(28), '#45536b'),
            (f"Group MAD norm {row['mad_normalized_reward']:+.3f}", font(29), '#45536b'),
        ]
        for line, face, color in lines:
            assert draw.textbbox((0, 0), line, font=face)[2] <= cell_width-36
            draw.text((label_x, label_y), line, font=face, fill=color)
            label_y += 41
    output, preview = Path(cfg['output']), Path(cfg['preview'])
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists() and not preview.exists(), 'Use a new versioned filename'
    sheet.save(output, format='PNG', compress_level=6)
    with Image.open(output) as saved:
        for image, item in zip(images, manifest):
            assert ImageChops.difference(saved.crop(item['tile_box']), image).getbbox() is None
    small = sheet.copy()
    preview_width = cfg.get('preview_width', 2140)
    small.thumbnail((preview_width, round(height*preview_width/width)), Image.Resampling.LANCZOS)
    small.save(preview, quality=93, subsampling=0)
    layout = dict(output=str(output), size=[width, height], preview=str(preview),
                  preview_size=list(small.size), original_tiles_pixel_identical=True,
                  group_mean=report['group_mean'], group_mad=report['group_mad'], tiles=manifest)
    output.with_suffix('.layout.json').write_text(json.dumps(layout, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), size=[width,height], bytes=output.stat().st_size,
                         preview=str(preview), preview_size=list(small.size),
                         original_tiles_pixel_identical=True)))


if __name__ == '__main__':
    main()
