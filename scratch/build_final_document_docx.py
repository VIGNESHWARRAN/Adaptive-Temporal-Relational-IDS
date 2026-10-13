"""Converts Adaptive_Temporal_Relational_IDS_Comprehensive_Master_Report.md into final_document.docx with professional styling and tables.
"""

import os
import re
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets cell background color."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    """Sets subtle light grey borders for table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="1A365D"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_docx_from_markdown(md_path, docx_path):
    doc = Document()

    # Set margins to 1 inch
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_code_block = False
    code_block_lines = []
    code_block_type = ""
    table_lines = []

    def flush_table(t_lines):
        if not t_lines:
            return
        # Parse Markdown table
        rows_data = []
        for line in t_lines:
            if re.match(r'^\s*\|?\s*:?-+:?\s*\|', line):
                continue # Skip delimiter row
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if any(cells):
                rows_data.append(cells)

        if not rows_data:
            return

        num_rows = len(rows_data)
        num_cols = max(len(r) for r in rows_data)

        tbl = doc.add_table(rows=num_rows, cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl)

        for r_idx, r_data in enumerate(rows_data):
            row = tbl.rows[r_idx]
            is_header = (r_idx == 0)

            for c_idx in range(num_cols):
                val = r_data[c_idx] if c_idx < len(r_data) else ""
                cell = row.cells[c_idx]
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_margins(cell, top=120, bottom=120, left=150, right=150)

                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)

                # Process bold / inline math in text
                clean_val = re.sub(r'\*\*(.*?)\*\*', r'\1', val)
                clean_val = clean_val.replace('$', '')
                
                run = p.add_run(clean_val)
                run.font.name = 'Calibri'

                if is_header:
                    set_cell_background(cell, "1A365D") # Dark navy header
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.font.size = Pt(10)
                else:
                    if r_idx % 2 == 1:
                        set_cell_background(cell, "F7FAFC") # Soft light grey row
                    else:
                        set_cell_background(cell, "FFFFFF")
                    
                    if "**" in val or val.startswith("0.") or val.endswith("%"):
                        if val.startswith("0.") or val.endswith("%"):
                            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    run.font.size = Pt(9.5)

        # Add spacing after table
        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(0)
        p_after.paragraph_format.space_after = Pt(6)

    def flush_code_block(c_lines, c_type):
        if not c_lines:
            return
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.right_indent = Inches(0.2)

        # Shaded background box for code/diagrams
        code_text = "".join(c_lines)
        if c_type == "mermaid":
            title_run = p.add_run("[Mermaid Architecture Flowchart Diagram]\n")
            title_run.font.bold = True
            title_run.font.size = Pt(10)
            title_run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
        
        run = p.add_run(code_text)
        run.font.name = 'Consolas'
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0) if c_type == "mermaid" else RGBColor(0x2D, 0x37, 0x48)

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        raw_line = line.rstrip('\n')

        # Code block handling
        if raw_line.startswith("```"):
            if in_code_block:
                flush_code_block(code_block_lines, code_block_type)
                code_block_lines = []
                in_code_block = False
            else:
                if table_lines:
                    flush_table(table_lines)
                    table_lines = []
                in_code_block = True
                code_block_type = raw_line.replace("```", "").strip()
            idx += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            idx += 1
            continue

        # Table handling
        if '|' in raw_line and not raw_line.startswith("```"):
            table_lines.append(raw_line)
            idx += 1
            continue
        else:
            if table_lines:
                flush_table(table_lines)
                table_lines = []

        # Empty lines
        if not raw_line.strip():
            idx += 1
            continue

        # Headings
        if raw_line.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(raw_line[2:].strip())
            run.font.name = 'Calibri'
            run.font.size = Pt(20)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D) # Deep Navy
        elif raw_line.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(raw_line[3:].strip())
            run.font.name = 'Calibri'
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0) # Slate Blue
        elif raw_line.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(raw_line[4:].strip())
            run.font.name = 'Calibri'
            run.font.size = Pt(12.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2C, 0x52, 0x82)
        elif raw_line.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(raw_line[5:].strip())
            run.font.name = 'Calibri'
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
        else:
            # Standard Paragraph
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15

            # Handle bullet items
            content = raw_line
            if content.startswith("* ") or content.startswith("- "):
                p.paragraph_format.left_indent = Inches(0.25)
                bullet_run = p.add_run("• ")
                bullet_run.font.bold = True
                bullet_run.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
                content = content[2:]

            # Parse inline formatting (**bold**, `code`, $math$)
            tokens = re.split(r'(\*\*.*?\*\*|`.*?`|\$.*?\$)', content)
            for token in tokens:
                if not token:
                    continue
                if token.startswith("**") and token.endswith("**"):
                    r = p.add_run(token[2:-2])
                    r.font.bold = True
                elif token.startswith("`") and token.endswith("`"):
                    r = p.add_run(token[1:-1])
                    r.font.name = 'Consolas'
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = RGBColor(0x80, 0x5A, 0xD5) # Purple code
                elif token.startswith("$") and token.endswith("$"):
                    r = p.add_run(token.strip("$"))
                    r.font.italic = True
                    r.font.color.rgb = RGBColor(0x2C, 0x52, 0x82) # Math italic blue
                else:
                    r = p.add_run(token)

        idx += 1

    if table_lines:
        flush_table(table_lines)

    doc.save(docx_path)
    print(f"Successfully generated DOCX Master Report: {docx_path}")

if __name__ == "__main__":
    md_in = "d:\\Final Year Implementation\\Adaptive_Temporal_Relational_IDS_Comprehensive_Master_Report.md"
    docx_out = "d:\\Final Year Implementation\\final_document.docx"
    build_docx_from_markdown(md_in, docx_out)
