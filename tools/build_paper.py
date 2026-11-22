# -*- coding: utf-8 -*-
"""Emit paper.tex (LaTeX source) and paper.docx (print pipeline → PDF)."""

import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper")
os.makedirs(PAPER, exist_ok=True)

sys.path.insert(0, HERE)
import paper_content as P  # noqa: E402

CN = "宋体"
CN_H = "黑体"
LATIN = "Times New Roman"


def esc(s):
    out = []
    for ch in s:
        if ch in "\\%&#$_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


# ----------------------------------------------------------------- LaTeX
def emit_tex():
    L = []
    L.append(r"\documentclass[10.5pt,a4paper]{ctexart}")
    L.append(r"\usepackage[margin=2.2cm]{geometry}")
    L.append(r"\usepackage{graphicx}")
    L.append(r"\usepackage{booktabs}")
    L.append(r"\usepackage{amsmath}")
    L.append(r"\usepackage[hidelinks]{hyperref}")
    L.append(r"\usepackage{caption}")
    L.append(r"\captionsetup{font=small,labelfont=bf}")
    L.append(r"\linespread{1.25}")
    L.append(r"\title{\bfseries %s}" % esc(P.TITLE))
    L.append(r"\author{%s\\[4pt]\small %s}" % (esc(P.AUTHORS), esc(P.AFFIL)))
    L.append(r"\date{}")
    L.append(r"\begin{document}")
    L.append(r"\maketitle")
    L.append(r"\begin{abstract}")
    L.append(r"\noindent %s" % esc(P.ABS_CN))
    L.append(r"\vspace{4pt}\noindent\textbf{关键词：}%s" % esc(P.KEYWORDS))
    L.append(r"\vspace{6pt}\noindent\textbf{Abstract.} %s" % esc(P.ABS_EN))
    L.append(r"\end{abstract}")
    fig_no = 0
    tab_no = 0
    for title, items in P.SECTIONS:
        L.append(r"\section*{%s}" % esc(title))
        L.append(r"\addcontentsline{toc}{section}{%s}" % esc(title))
        for item in items:
            kind = item[0]
            if kind == "p":
                L.append(esc(item[1]))
                L.append("")
            elif kind == "fig":
                fig_no += 1
                L.append(r"\begin{figure}[htbp]")
                L.append(r"\centering")
                L.append(r"\includegraphics[width=0.9\textwidth]{%s}" % ("../" + item[1]).replace("\\", "/"))
                L.append(r"\caption{%s}" % esc(item[2]))
                L.append(r"\end{figure}")
            elif kind == "table":
                tab_no += 1
                d = item[1]
                cols = len(d["header"])
                spec = "l" + "c" * (cols - 1)
                L.append(r"\begin{table}[htbp]")
                L.append(r"\centering")
                L.append(r"\caption{%s}" % esc(d["caption"]))
                L.append(r"\begin{tabular}{%s}" % spec)
                L.append(r"\toprule")
                L.append(" & ".join(esc(h) for h in d["header"]) + r" \\")
                L.append(r"\midrule")
                for row in d["rows"]:
                    L.append(" & ".join(esc(c) for c in row) + r" \\")
                L.append(r"\bottomrule")
                L.append(r"\end{tabular}")
                L.append(r"\end{table}")
    L.append(r"\section*{参考文献}")
    L.append(r"\begin{enumerate}")
    for r in P.REFS:
        L.append(r"  \item %s" % esc(r))
    L.append(r"\end{enumerate}")
    L.append(r"\end{document}")
    path = os.path.join(PAPER, "paper.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


# ----------------------------------------------------------------- DOCX
def style_run(run, size=None, bold=None, font=CN, color=None, italic=None):
    run.font.name = LATIN
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), font)
    rFonts.set(qn("w:ascii"), LATIN)
    rFonts.set(qn("w:hAnsi"), LATIN)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color:
        run.font.color.rgb = color


def shade(cell, hex_fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    cell._element.get_or_add_tcPr().append(shd)


def emit_docx():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.2)

    st = doc.styles["Normal"]
    st.font.name = LATIN
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.line_spacing = 1.32

    # title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    style_run(p.add_run(P.TITLE), size=17, bold=True, font=CN_H)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    style_run(p.add_run(P.AUTHORS), size=11.5, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    style_run(p.add_run(P.AUTHORS_NOTE), size=9, italic=True, color=RGBColor(0x55, 0x55, 0x55))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    style_run(p.add_run(P.AFFIL), size=9, color=RGBColor(0x55, 0x55, 0x55))

    # abstract
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.right_indent = Cm(0.9)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    style_run(p.add_run("摘要　"), size=9.5, bold=True, font=CN_H)
    style_run(p.add_run(P.ABS_CN), size=9.5)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.right_indent = Cm(0.9)
    style_run(p.add_run("关键词　"), size=9.5, bold=True, font=CN_H)
    style_run(p.add_run(P.KEYWORDS), size=9.5)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.right_indent = Cm(0.9)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    style_run(p.add_run("Abstract　"), size=9, bold=True)
    style_run(p.add_run(P.ABS_EN), size=9, italic=True)

    def heading(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(11)
        h.paragraph_format.space_after = Pt(4)
        style_run(h.add_run(text), size=12.5, bold=True, font=CN_H)

    def body(text):
        b = doc.add_paragraph()
        b.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        b.paragraph_format.first_line_indent = Cm(0.74)
        style_run(b.add_run(text), size=10.5)

    fig_no = 0
    tab_no = 0
    for title, items in P.SECTIONS:
        heading(title)
        for item in items:
            kind = item[0]
            if kind == "p":
                body(item[1])
            elif kind == "fig":
                fig_no += 1
                path = os.path.join(ROOT, item[1])
                pic = doc.add_paragraph()
                pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pic.paragraph_format.space_before = Pt(6)
                pic.paragraph_format.space_after = Pt(2)
                if os.path.exists(path):
                    pic.add_run().add_picture(path, width=Cm(15.2))
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap.paragraph_format.space_after = Pt(8)
                style_run(cap.add_run(item[2]), size=8.5, color=RGBColor(0x33, 0x33, 0x33))
            elif kind == "table":
                tab_no += 1
                d = item[1]
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap.paragraph_format.space_before = Pt(6)
                cap.paragraph_format.space_after = Pt(3)
                style_run(cap.add_run(d["caption"]), size=8.5, color=RGBColor(0x33, 0x33, 0x33))
                ncol = len(d["header"])
                tbl = doc.add_table(rows=1 + len(d["rows"]), cols=ncol)
                tbl.style = "Table Grid"
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                for ci, h in enumerate(d["header"]):
                    cell = tbl.cell(0, ci)
                    cell.text = ""
                    par = cell.paragraphs[0]
                    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    style_run(par.add_run(h), size=8.5, bold=True)
                    shade(cell, "EDF1F7")
                for ri, row in enumerate(d["rows"], start=1):
                    for ci, val in enumerate(row):
                        cell = tbl.cell(ri, ci)
                        cell.text = ""
                        par = cell.paragraphs[0]
                        par.alignment = WD_ALIGN_PARAGRAPH.CENTER if ci else WD_ALIGN_PARAGRAPH.LEFT
                        style_run(par.add_run(val), size=8.5)
                doc.add_paragraph().paragraph_format.space_after = Pt(2)

    heading("参考文献")
    for i, ref in enumerate(P.REFS, start=1):
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Cm(0.8)
        par.paragraph_format.first_line_indent = Cm(-0.8)
        par.paragraph_format.space_after = Pt(2)
        style_run(par.add_run("[%d] %s" % (i, ref)), size=9)

    out = os.path.join(PAPER, "paper.docx")
    doc.save(out)
    return out


if __name__ == "__main__":
    tex = emit_tex()
    dx = emit_docx()
    print("tex :", tex, os.path.getsize(tex))
    print("docx:", dx, os.path.getsize(dx))
