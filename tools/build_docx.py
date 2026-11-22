import os
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "report", "report.md")
OUT = os.path.join(ROOT, "report", "report.docx")

CN = "微软雅黑"
LATIN = "Calibri"
MONO = "Consolas"


def style_run(run, size=None, bold=None, mono=False, color=None):
    run.font.name = MONO if mono else LATIN
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), CN)
    rFonts.set(qn("w:ascii"), MONO if mono else LATIN)
    rFonts.set(qn("w:hAnsi"), MONO if mono else LATIN)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_inline(par, text, size=None, base_bold=False):
    """Render **bold**, `code`, *italic* inline spans."""
    tokens = re.split(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = par.add_run(tok[2:-2])
            style_run(r, size=size, bold=True)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            style_run(r, size=(size - 1) if size else 9.5, mono=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            style_run(r, size=size, bold=base_bold)
            r.italic = True
        else:
            r = par.add_run(tok)
            style_run(r, size=size, bold=base_bold)


def shade(cell_or_par, hex_color):
    el = cell_or_par._element
    pr = el.find(qn("w:tcPr")) if el.tag.endswith("tc") else el
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    if el.tag.endswith("tc"):
        pr.append(shd)
    else:
        pr.append(shd)


def set_cell_text(cell, text, bold=False, size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    add_inline(p, text, size=size, base_bold=bold)


doc = Document()

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(2.0)

normal = doc.styles["Normal"]
normal.font.name = LATIN
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.25

for lvl, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
    st = doc.styles[lvl]
    st.font.name = LATIN
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
    st.paragraph_format.space_before = Pt(12)
    st.paragraph_format.space_after = Pt(5)

lines = open(SRC, encoding="utf-8-sig").read().splitlines()

i = 0
n_tables = 0
n_paras = 0

# ---------- cover ----------
assert lines[0].startswith("# ")
cover_title = lines[0][2:].strip()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(90)
r = p.add_run(cover_title)
style_run(r, size=26, bold=True, color=RGBColor(0x1F, 0x3B, 0x57))

meta = []
i = 1
while i < len(lines) and not lines[i].startswith("## "):
    s = lines[i].strip()
    if s and s != "---":
        meta.append(s)
    i += 1

for m in meta:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    add_inline(p, m.replace("**", ""), size=10.5)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(40)
r = p.add_run("— 交付物：完整实验报告 + 实验数据包（1.8 GB，3 个 tar.gz）—")
style_run(r, size=9, color=RGBColor(0x66, 0x66, 0x66))

doc.add_page_break()

# ---------- body ----------
while i < len(lines):
    line = lines[i]
    s = line.strip()

    if not s or s == "---":
        i += 1
        continue

    # code fence
    if s.startswith("```"):
        i += 1
        block = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            block.append(lines[i])
            i += 1
        i += 1
        for bl in block:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.left_indent = Cm(0.4)
            r = p.add_run(bl if bl else " ")
            style_run(r, size=9, mono=True)
            shade(p, "F2F4F7")
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        continue

    # table
    if s.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c or "---") for c in cells):
                rows.append(cells)
            i += 1
        if rows:
            ncol = max(len(r) for r in rows)
            tbl = doc.add_table(rows=len(rows), cols=ncol)
            tbl.style = "Table Grid"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            tbl.autofit = True
            for ri, row in enumerate(rows):
                for ci in range(ncol):
                    txt = row[ci] if ci < len(row) else ""
                    cell = tbl.cell(ri, ci)
                    set_cell_text(cell, txt, bold=(ri == 0), size=8.5)
                    if ri == 0:
                        shade(cell, "DCE6F1")
            n_tables += 1
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
        continue

    # image
    m_img = re.match(r"^!\[(.*?)\]\((.+?)\)$", s)
    if m_img:
        alt, path = m_img.group(1), m_img.group(2)
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(SRC), path)
        if os.path.exists(path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.add_run().add_picture(path, width=Cm(16.4))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(cap, alt, size=8.5)
            n_paras += 1
        i += 1
        continue

    # headings
    m = re.match(r"^(#{1,4})\s+(.*)$", s)
    if m:
        level = len(m.group(1))
        text = m.group(2).strip()
        if level == 1:
            doc.add_page_break()
            p = doc.add_paragraph()
            r = p.add_run(text)
            style_run(r, size=20, bold=True, color=RGBColor(0x1F, 0x3B, 0x57))
        elif level == 2:
            h = doc.add_heading(level=1)
            add_inline(h, text, size=16)
        elif level == 3:
            h = doc.add_heading(level=2)
            add_inline(h, text, size=13)
        else:
            h = doc.add_heading(level=3)
            add_inline(h, text, size=11.5)
        n_paras += 1
        i += 1
        continue

    # bullets
    m = re.match(r"^-\s+(.*)$", s)
    if m:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        add_inline(p, m.group(1), size=10.5)
        n_paras += 1
        i += 1
        continue

    m = re.match(r"^(\d+)\.\s+(.*)$", s)
    if m:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(2)
        add_inline(p, m.group(2), size=10.5)
        n_paras += 1
        i += 1
        continue

    # blockquote / italic note
    if s.startswith(">"):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        add_inline(p, s.lstrip("> ").strip(), size=9.5)
        n_paras += 1
        i += 1
        continue

    # normal paragraph (merge consecutive lines)
    buf = [s]
    i += 1
    while i < len(lines):
        nxt = lines[i].strip()
        if (not nxt) or nxt.startswith(("#", "|", "```", "- ", ">")) or re.match(r"^\d+\.\s", nxt) or nxt == "---":
            break
        buf.append(nxt)
        i += 1
    p = doc.add_paragraph()
    add_inline(p, " ".join(buf), size=10.5)
    n_paras += 1

doc.save(OUT)

chk = Document(OUT)
print("saved:", OUT)
print("paragraphs:", len(chk.paragraphs), "tables:", len(chk.tables), "n_tables_expected:", n_tables)
print("first paragraphs:", [p.text[:40] for p in chk.paragraphs[:6]])
