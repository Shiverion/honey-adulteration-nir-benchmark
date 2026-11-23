"""Render the journal-neutral manuscript Markdown into an editable Word file."""

import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor


directory = Path(__file__).resolve().parent
source = directory / "manuscript.md"
document = Document()
section = document.sections[0]
section.page_width, section.page_height = Inches(8.27), Inches(11.69)
section.left_margin = section.right_margin = Inches(.8)
section.top_margin = section.bottom_margin = Inches(.79)
section.footer_distance = Inches(.36)


def configure(name, size, before=0, after=5, bold=False):
    if name not in document.styles:
        document.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style = document.styles[name]
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)
    return style


normal = configure("Normal", 11, after=7)
normal.paragraph_format.line_spacing = 1.12
normal.paragraph_format.widow_control = True
title = configure("Title", 24, after=13, bold=True)
title.paragraph_format.line_spacing = 1.06
title.paragraph_format.keep_with_next = True
title_ppr = title.element.get_or_add_pPr()
for border in title_ppr.findall(qn("w:pBdr")):
    title_ppr.remove(border)
for name, size in (("Heading 1", 15), ("Heading 2", 12.5)):
    heading = configure(name, size, before=10, after=4, bold=True)
    heading.paragraph_format.keep_with_next = True
caption = configure("Caption", 9.5, before=4, after=8)
caption.paragraph_format.line_spacing = 1.10
cell_style = configure("Table body", 9.5, after=0)
cell_style.paragraph_format.line_spacing = 1.05
reference = configure("Reference entry", 10, after=6)
reference.paragraph_format.left_indent = Inches(.24)
reference.paragraph_format.first_line_indent = Inches(-.24)

props = document.core_properties
props.title = "Comparative evaluation of near infrared models for honey adulteration detection"
props.author = ""
props.keywords = "honey, near-infrared spectroscopy, adulteration, grouped validation"
props.subject = "Research article"
props.comments = ""
footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def add_runs(paragraph, content):
    for token in re.split(r"(\*\*[^*]+\*\*|https://[^\s]+)", content):
        if token.startswith("https://"):
            hyperlink = OxmlElement("w:hyperlink")
            hyperlink.set(qn("r:id"), paragraph.part.relate_to(token, RT.HYPERLINK, is_external=True))
            run = OxmlElement("w:r")
            text = OxmlElement("w:t")
            text.text = token
            run.append(text)
            hyperlink.append(run)
            paragraph._p.append(hyperlink)
        elif token.startswith("**") and token.endswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        elif token:
            paragraph.add_run(token)


def table_cells(line):
    return [text.strip() for text in line.strip().strip("|").split("|")]


def add_table(block):
    parsed = [table_cells(line) for line in block.splitlines()]
    header, alignments, *body = parsed
    table = document.add_table(rows=0, cols=len(header))
    table.autofit = False
    column_weights = {
        "Release": (.15, .26, .25, .34), "Model and comparator": (.32, .14, .23, .11, .11, .09),
    }
    weights = column_weights.get(header[0], (1.25, 1.38, 1, 1, .85, .92))
    width = 6.58
    for column, weight in zip(table.columns, weights):
        column.width = Inches(width * weight / sum(weights))
    properties = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "5")
        edge.set(qn("w:color"), "D9D9D9")
        borders.append(edge)
    properties.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for side, dimension in (("top", 50), ("bottom", 50), ("left", 90), ("right", 90)):
        margin = OxmlElement(f"w:{side}")
        margin.set(qn("w:w"), str(dimension))
        margin.set(qn("w:type"), "dxa")
        margins.append(margin)
    properties.append(margins)
    for i, values in enumerate([header, *body]):
        row = table.add_row()
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for j, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = table.columns[j].width
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.style = cell_style
            p.paragraph_format.keep_with_next = i < len(body)
            if alignments[j].endswith(":"):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p.add_run(value).bold = i == 0
            if i == 0:
                color = OxmlElement("w:shd")
                color.set(qn("w:fill"), "E7EBF0")
                cell._tc.get_or_add_tcPr().append(color)


for block in re.split(r"\n\s*\n", source.read_text(encoding="utf-8").strip()):
    figure = re.fullmatch(r"!\[(.+)\]\((.+)\)", block)
    if figure:
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        shape = p.add_run().add_picture(str(directory / figure[2]), width=Inches(6.53))
        shape._inline.docPr.set("descr", figure[1])
    elif block.startswith("|"):
        add_table(block)
    elif block.startswith("# "):
        document.add_paragraph(block[2:], "Title")
    elif block.startswith("## "):
        document.add_paragraph(block[3:], "Heading 1")
    elif block.startswith("### "):
        document.add_paragraph(block[4:], "Heading 2")
    else:
        label = re.match(r"^(Table|Figure) \d+\.", block)
        is_reference = re.match(r"^\[\d+\] ", block) is not None
        style = caption if label else reference if is_reference else normal
        p = document.add_paragraph(style=style)
        if label and label[1] == "Table":
            p.paragraph_format.keep_with_next = True
        add_runs(p, block.replace("\n", " "))

document.save(directory / "honey_adulteration_manuscript.docx")
