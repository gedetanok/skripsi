#!/usr/bin/env python3
# =================================================================
# Membangun body_docx*.md (front matter + isi) dan reference.docx
# untuk konversi proposal ke Word yang TIDAK hancur strukturnya.
#
# Dwibahasa:  python3 _docx_build.py id   (default)
#             python3 _docx_build.py en
#
#   - front matter (raw OpenXML, dari metadata*.yaml): Sampul, Lembar
#     Persetujuan, Abstrak, dan Daftar Isi otomatis (field TOC Word).
#     Versi id memuat ABSTRAK (ID) + ABSTRACT (EN); versi en hanya ABSTRACT.
#   - isi: preambel judul dibuang; tiap bab "## X" diberi label
#     "BAB/CHAPTER <Romawi> X". Dengan --shift-heading-level-by=-1 di
#     pandoc, bab menjadi Heading 1, subbab Heading 2/3 (outline benar).
#   - reference.docx: Times New Roman 12, A4, margin kiri 4 cm /
#     atas-kanan-bawah 3 cm, spasi ganda (Pedoman Undiksha).
# Dipanggil oleh build_docx.sh / build_docx_en.sh.
# =================================================================
import os, re, io, sys, zipfile, subprocess
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
REF = os.path.join(SCRIPT_DIR, "reference.docx")
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

LANG = sys.argv[1] if len(sys.argv) > 1 else "id"

CFG = {
    "id": {
        "src": os.path.join(REPO_DIR, "draft_proposal_skripsi.md"),
        "meta": os.path.join(SCRIPT_DIR, "metadata.yaml"),
        "body": os.path.join(SCRIPT_DIR, "body_docx.md"),
        "figdir": "figures",
        "chapter": "BAB", "fig": "Gambar", "tbl": "Tabel",
        "dual_abstract": True,
        "L": {
            "proposal": "PROPOSAL SKRIPSI", "by": "Oleh:", "nim_label": "NIM. ",
            "prog_prefix": "PROGRAM STUDI ", "dept_prefix": "JURUSAN ", "fac_prefix": "FAKULTAS ",
            "approval_title": "LEMBAR PERSETUJUAN PROPOSAL SKRIPSI",
            "entitled": "Proposal skripsi dengan judul:", "prepared_by": "yang disusun oleh:",
            "name": "Nama", "nim_row": "NIM", "prog_row": "Program Studi",
            "dept_row": "Jurusan", "fac_row": "Fakultas",
            "approved": "telah disetujui oleh dosen pembimbing untuk diajukan dalam seminar proposal skripsi.",
            "sup1": "Pembimbing I", "sup2": "Pembimbing II", "nip": "NIP. ",
            "proposal_kind": "Proposal Skripsi",
            "toc_title": "DAFTAR ISI",
            "toc_note": "Klik kanan daftar isi ini, lalu pilih Update Field, untuk menampilkan judul bab beserta nomor halaman.",
        },
    },
    "en": {
        "src": os.path.join(REPO_DIR, "draft_proposal_skripsi_en.md"),
        "meta": os.path.join(SCRIPT_DIR, "metadata_en.yaml"),
        "body": os.path.join(SCRIPT_DIR, "body_docx_en.md"),
        "figdir": "figures_en",
        "chapter": "CHAPTER", "fig": "Figure", "tbl": "Table",
        "dual_abstract": False,
        "L": {
            "proposal": "THESIS PROPOSAL", "by": "By:", "nim_label": "Student ID Number: ",
            "prog_prefix": "STUDY PROGRAM OF ", "dept_prefix": "DEPARTMENT OF ", "fac_prefix": "FACULTY OF ",
            "approval_title": "THESIS PROPOSAL APPROVAL SHEET",
            "entitled": "The thesis proposal entitled:", "prepared_by": "prepared by:",
            "name": "Name", "nim_row": "Student ID Number", "prog_row": "Study Program",
            "dept_row": "Department", "fac_row": "Faculty",
            "approved": "has been approved by the supervisors to be presented in the thesis proposal seminar.",
            "sup1": "Supervisor I", "sup2": "Supervisor II", "nip": "NIP. ",
            "proposal_kind": "Thesis Proposal",
            "toc_title": "TABLE OF CONTENTS",
            "toc_note": "Right-click this table of contents and choose Update Field to display chapter titles and page numbers.",
        },
    },
}
cfg = CFG[LANG]


# ----------------------------------------------------------------
# 1. reference.docx: font, ukuran kertas, margin, spasi
# ----------------------------------------------------------------
def make_reference(ref_path):
    raw = subprocess.run(
        ["pandoc", "--print-default-data-file", "reference.docx"],
        capture_output=True, check=True).stdout
    zin = zipfile.ZipFile(io.BytesIO(raw))
    items = {n: zin.read(n) for n in zin.namelist()}

    th = "word/theme/theme1.xml"
    if th in items:
        t = items[th].decode("utf-8")
        t = re.sub(r'(<a:majorFont>\s*<a:latin[^>]*?typeface=")[^"]*(")', r"\1Times New Roman\2", t)
        t = re.sub(r'(<a:minorFont>\s*<a:latin[^>]*?typeface=")[^"]*(")', r"\1Times New Roman\2", t)
        items[th] = t.encode("utf-8")

    st = "word/styles.xml"
    s = items[st].decode("utf-8")
    s = s.replace(
        '<w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia" w:hAnsiTheme="minorHAnsi" w:cstheme="minorBidi" />',
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" />')
    s = s.replace('<w:spacing w:after="200" />',
                  '<w:spacing w:after="0" w:line="480" w:lineRule="auto" /><w:ind w:firstLine="720" />')
    # Pandoc's "First Paragraph" style zeroes the first-line indent on the first
    # paragraph after a heading; neutralise it so every paragraph indents.
    s = re.sub(r'(<w:style [^>]*w:styleId="FirstParagraph".*?)<w:ind\b[^/]*/>',
               r'\1', s, flags=re.S)

    def resize(style_id, sz):
        def repl(m):
            b = m.group(0)
            b = re.sub(r'<w:sz w:val="\d+" />', f'<w:sz w:val="{sz}" />', b)
            b = re.sub(r'<w:szCs w:val="\d+" />', f'<w:szCs w:val="{sz}" />', b)
            return b
        return re.sub(r'<w:style [^>]*w:styleId="' + style_id + r'".*?</w:style>',
                      repl, s, flags=re.S)
    s = resize("Heading1", 28)
    s = resize("Heading2", 26)
    s = resize("Heading3", 24)
    items[st] = s.encode("utf-8")

    dx = "word/document.xml"
    d = items[dx].decode("utf-8")
    new_sect = ('<w:sectPr>'
                '<w:footnotePr><w:numRestart w:val="eachSect" /></w:footnotePr>'
                '<w:pgSz w:w="11906" w:h="16838" />'
                '<w:pgMar w:top="1701" w:right="1701" w:bottom="1701" w:left="2268" '
                'w:header="708" w:footer="708" w:gutter="0" />'
                '</w:sectPr>')
    d = re.sub(r"<w:sectPr>.*?</w:sectPr>", lambda m: new_sect, d, flags=re.S)
    items[dx] = d.encode("utf-8")

    with zipfile.ZipFile(ref_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, data in items.items():
            zout.writestr(n, data)


# ----------------------------------------------------------------
# 2. Front matter sebagai raw OpenXML
# ----------------------------------------------------------------
def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def collapse(t):
    return re.sub(r"\s+", " ", str(t)).strip()

def para(text="", align=None, bold=False, italic=False, sz=None, style=None):
    ppr = "<w:pPr>"
    if style:
        ppr += f'<w:pStyle w:val="{style}" />'
    ppr += '<w:spacing w:after="0" w:line="240" w:lineRule="auto" />'  # front matter: 1 spasi
    if align:
        ppr += f'<w:jc w:val="{align}" />'
    ppr += "</w:pPr>"
    parts = ""
    if bold:
        parts += "<w:b />"
    if italic:
        parts += "<w:i />"
    if sz:
        parts += f'<w:sz w:val="{sz}" /><w:szCs w:val="{sz}" />'
    rpr = "<w:rPr>" + parts + "</w:rPr>" if parts else ""
    return f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'

def kvrow(label, value):
    ppr = ('<w:pPr><w:tabs><w:tab w:val="left" w:pos="2700" /></w:tabs>'
           '<w:spacing w:after="0" w:line="240" w:lineRule="auto" /></w:pPr>')
    return (f'<w:p>{ppr}'
            f'<w:r><w:t xml:space="preserve">{esc(label)}</w:t></w:r>'
            f'<w:r><w:tab /></w:r>'
            f'<w:r><w:t xml:space="preserve">: {esc(value)}</w:t></w:r></w:p>')

def gap():
    return '<w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto" /></w:pPr></w:p>'

def pagebreak():
    return '<w:p><w:r><w:br w:type="page" /></w:r></w:p>'

def toc_field(note):
    instr = r' TOC \o "1-3" \h \z \u '
    return ('<w:p>'
            '<w:r><w:fldChar w:fldCharType="begin" /></w:r>'
            f'<w:r><w:instrText xml:space="preserve">{instr}</w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate" /></w:r>'
            f'<w:r><w:rPr><w:i /></w:rPr><w:t xml:space="preserve">{esc(note)}</w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="end" /></w:r>'
            '</w:p>')

def abstract_block(title_label, kind, sup_lead1, sup_lead2, kw_label, body_text, kw_text, m):
    out = [para(title_label, style="Heading1", align="center")]
    out.append(para(f'{m["author"]}. {m["nim"]}. {m["tahun"]}. {collapse(m["thesis-title"])}. '
                    f'{kind}. {m["program-studi"]}, {m["jurusan"]}, {m["fakultas"]}, {m["universitas"]}.',
                    italic=True))
    out.append(gap())
    out.append(para(f'{sup_lead1}{m["pembimbing1-nama"]}; {sup_lead2}{m["pembimbing2-nama"]}'))
    out.append(gap())
    out.append(para(collapse(body_text), align="both"))
    out.append(gap())
    out.append(para(kw_label + collapse(kw_text), bold=True))
    out.append(pagebreak())
    return out

def front_matter(m, cfg):
    L = cfg["L"]
    title = collapse(m["thesis-title"])
    fm = []
    # ---- Sampul / Cover ----
    fm += [gap(), gap()]
    fm.append(para(L["proposal"], align="center", bold=True, sz=28))
    fm.append(gap())
    fm.append(para(title.upper(), align="center", bold=True, sz=26))
    fm += [gap(), gap(), gap()]
    fm.append(para(L["by"], align="center"))
    fm.append(para(m["author"], align="center", bold=True))
    fm.append(para(L["nim_label"] + str(m["nim"]), align="center"))
    fm += [gap(), gap(), gap()]
    fm.append(para(L["prog_prefix"] + str(m["program-studi"]).upper(), align="center", bold=True))
    fm.append(para(L["dept_prefix"] + str(m["jurusan"]).upper(), align="center", bold=True))
    fm.append(para(L["fac_prefix"] + str(m["fakultas"]).upper(), align="center", bold=True))
    fm.append(para(str(m["universitas"]).upper(), align="center", bold=True))
    fm.append(para(str(m["tahun"]), align="center", bold=True))
    fm.append(pagebreak())
    # ---- Lembar Persetujuan / Approval Sheet ----
    fm.append(para(L["approval_title"], align="center", bold=True))
    fm.append(gap())
    fm.append(para(L["entitled"]))
    fm.append(para(title, align="center", bold=True))
    fm.append(gap())
    fm.append(para(L["prepared_by"]))
    fm.append(kvrow(L["name"], m["author"]))
    fm.append(kvrow(L["nim_row"], m["nim"]))
    fm.append(kvrow(L["prog_row"], m["program-studi"]))
    fm.append(kvrow(L["dept_row"], m["jurusan"]))
    fm.append(kvrow(L["fac_row"], m["fakultas"]))
    fm.append(gap())
    fm.append(para(L["approved"]))
    fm += [gap(), gap()]
    fm.append(para(str(m["kota"]) + ", " + str(m["tahun"]), align="right"))
    fm += [gap(), gap()]
    fm.append(para(L["sup1"]))
    fm += [gap(), gap(), gap()]
    fm.append(para(str(m["pembimbing1-nama"]), bold=True))
    fm.append(para(L["nip"] + str(m["pembimbing1-nip"])))
    fm.append(gap())
    fm.append(para(L["sup2"]))
    fm += [gap(), gap(), gap()]
    fm.append(para(str(m["pembimbing2-nama"]), bold=True))
    fm.append(para(L["nip"] + str(m["pembimbing2-nip"])))
    fm.append(pagebreak())
    # ---- Abstrak (ID) bila dwibahasa ----
    if cfg["dual_abstract"]:
        fm += abstract_block("ABSTRAK", "Proposal Skripsi",
                             "Pembimbing I: ", "Pembimbing II: ", "Kata Kunci: ",
                             m["abstrak-id"], m["kata-kunci-id"], m)
    # ---- Abstract (EN) ----
    fm += abstract_block("ABSTRACT", L["proposal_kind"],
                         "Supervisor I: ", "Supervisor II: ", "Keywords: ",
                         m["abstract-en"], m["keywords-en"], m)
    # ---- Daftar Isi / Table of Contents (field TOC otomatis) ----
    fm.append(para(L["toc_title"], align="center", bold=True))
    fm.append(gap())
    fm.append(toc_field(L["toc_note"]))
    fm.append(pagebreak())
    return "\n```{=openxml}\n" + "\n".join(fm) + "\n```\n\n"


# ----------------------------------------------------------------
# 3. Isi: buang preambel, beri label bab, nomori Gambar/Tabel, \refitem
# ----------------------------------------------------------------
def braced(text, cmd):
    i = text.find(cmd + "{")
    if i < 0:
        return ""
    j = i + len(cmd) + 1
    depth = 1
    out = []
    while j < len(text) and depth:
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth:
            out.append(c)
        j += 1
    return "".join(out)

def cap_to_md(t):
    t = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", t)
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", t)
    t = t.replace(r"\&", "&").replace(r"\ldots", "...").replace("~", " ")
    t = re.sub(r"\\quad|\\,|\\!|\\ ", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def transform_body(src, chapter_word, fig_word, tbl_word, figdir):
    idx = src.find("\n## ")
    body = src[idx + 1:] if idx >= 0 else src

    cnt = [0]
    def label(m):
        head = m.group(1)
        if "{.unnumbered}" in head:
            return "## " + head
        cnt[0] += 1
        return f"## {chapter_word} {ROMAN[cnt[0] - 1]} {head}"
    body = re.sub(r"^## (.+)$", label, body, flags=re.M)

    numbered = [mm.start() for mm in re.finditer(r"^## +(.+)$", body, re.M)
                if "{.unnumbered}" not in mm.group(1)]
    def chapter_of(pos):
        return sum(1 for s in numbered if s <= pos)

    fig_cnt, tbl_cnt = {}, {}
    repls = []
    for m in re.finditer(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", body, re.DOTALL):
        env, kind = m.group(0), m.group(1)
        ch = chapter_of(m.start())
        cap = cap_to_md(braced(env, r"\caption"))
        if kind == "figure":
            inc = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", env)
            lab = re.search(r"\\label\{fig:([A-Za-z0-9_]+)\}", env)
            if inc:
                png = inc.group(1)
            elif lab:
                png = f"{figdir}/fig_{lab.group(1)}.png"
            else:
                continue
            fig_cnt[ch] = fig_cnt.get(ch, 0) + 1
            num = f"{fig_word} {ch}.{fig_cnt[ch]}"
        else:
            lab = re.search(r"\\label\{tab:([A-Za-z0-9_]+)\}", env)
            png = f"{figdir}/fig_{lab.group(1)}.png" if lab else f"{figdir}/fig_jadwal.png"
            tbl_cnt[ch] = tbl_cnt.get(ch, 0) + 1
            num = f"{tbl_word} {ch}.{tbl_cnt[ch]}"
        repls.append((m.start(), m.end(), f"![**{num}.** {cap}]({png})"))
    for s, e, txt in sorted(repls, reverse=True):
        body = body[:s] + txt + body[e:]

    def conv_ref(m):
        t = m.group(1)
        t = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", t)
        t = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", t)
        t = t.replace(r"\&", "&").replace(r"\ldots", "...").replace("~", " ")
        t = re.sub(r"\\\s", " ", t)
        return t.strip() + "\n"
    body = re.sub(r"\\refitem\{(.*?)\}\s*(?=\n\\refitem|\n\n|\Z)",
                  conv_ref, body, flags=re.DOTALL)
    return body


def main():
    with open(cfg["meta"], encoding="utf-8") as f:
        m = list(yaml.safe_load_all(f))[0]
    print(f"[docx:{LANG}] membuat reference.docx (Times New Roman 12, A4, margin, spasi ganda) ...")
    make_reference(REF)
    print(f"[docx:{LANG}] menyusun front matter + isi -> {os.path.basename(cfg['body'])} ...")
    src = open(cfg["src"], encoding="utf-8").read()
    out = front_matter(m, cfg) + transform_body(
        src, cfg["chapter"], cfg["fig"], cfg["tbl"], cfg["figdir"])
    open(cfg["body"], "w", encoding="utf-8").write(out)
    print("  ok")


if __name__ == "__main__":
    main()
