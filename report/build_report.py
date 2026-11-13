"""Build the 15-page report: report/report.md -> HTML (A4, 11 pt, 2.5 cm margins) -> PDF (Chromium)
and DOCX (pandoc). Numbers are written as {{name}} or {{name:fmt}} and resolved only through
results/manifest.json (tools/numbers.json names). The main text ends at the <!--END-MAIN--> marker;
the script reports the page on which references start, so the main text can be held at 15 pages.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "report"
CSS = """
@page { size: A4; margin: 2.5cm; }
html { font-family: 'DejaVu Serif', Georgia, serif; font-size: 11pt; line-height: 1.38; color: #111; }
body { margin: 0; }
h1 { font-size: 16pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 12pt 0 4pt; }
h3 { font-size: 11pt; margin: 8pt 0 2pt; }
p { margin: 0 0 5pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 5pt 0; padding-left: 16pt; }
li { margin: 0 0 2pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 4pt 0 8pt; page-break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #999; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 1pt solid #333; }
figure { margin: 6pt 0 8pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figcaption { font-size: 8.8pt; text-align: left; color: #222; margin-top: 2pt; }
.caption { font-size: 8.8pt; }
.abstract { border: 0.6pt solid #888; padding: 6pt 8pt; margin: 6pt 0 8pt; font-size: 10pt; }
.meta { font-size: 9.5pt; color: #333; }
.pb { page-break-before: always; }
code { font-size: 9pt; }
"""


def resolve(md: str) -> str:
    man = json.loads((ROOT / "results/manifest.json").read_text())["numbers"]

    def sub(m):
        name, fmt = m.group(1), m.group(2)
        if name not in man or man[name]["value"] is None:
            raise KeyError(f"number {name} not in manifest")
        v = man[name]["value"]
        if fmt:
            return format(float(v), fmt)
        return str(v)
    return re.sub(r"\{\{([A-Za-z0-9_.\-]+)(?::([^}]+))?\}\}", sub, md)


def main():
    md = resolve((R / "report.md").read_text())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    md = md.replace("@@COMMIT@@", commit)
    (R / "build").mkdir(exist_ok=True)
    (R / "build/report_resolved.md").write_text(md)
    html = subprocess.run(["pandoc", "-f", "markdown+pipe_tables+raw_html+implicit_figures", "-t", "html5", "--resource-path", str(ROOT)],
                          input=md, capture_output=True, text=True, check=True).stdout
    page = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    page = page.replace('src="results/', f'src="file://{ROOT}/results/').replace('src="proposal_inserts/', f'src="file://{ROOT}/proposal_inserts/')
    (R / "build/report.html").write_text(page)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(f"file://{R}/build/report.html")
        pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(R / "SAMA_insilico_report.pdf"), format="A4", margin={"top": "2.5cm", "bottom": "2.5cm", "left": "2.5cm", "right": "2.5cm"},
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8pt;width:100%;text-align:center'><span class='pageNumber'></span></div>")
        b.close()
    subprocess.run(["pandoc", str(R / "build/report_resolved.md"), "-o", str(R / "SAMA_insilico_report.docx"), "--resource-path", str(ROOT)], check=True)
    # where does the main text end?
    out = subprocess.run(["pdftotext", "-layout", str(R / "SAMA_insilico_report.pdf"), "-"], capture_output=True, text=True).stdout
    pages = out.split("\f")
    ref_page = next((i + 1 for i, t in enumerate(pages) if re.search(r"^\s*References\s*$", t, re.M)), None)
    print(f"pages: {len(pages) - 1 if pages[-1].strip() == '' else len(pages)}; references start on page {ref_page}; main text = {ref_page - 1 if ref_page else '?'} pages")
    if ref_page:
        last = pages[ref_page - 2]
        print(f"lines on last main page: {len([ln for ln in last.splitlines() if ln.strip()])}")


if __name__ == "__main__":
    sys.exit(main())
