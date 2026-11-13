"""Build the proposal documents (PDF + DOCX) with every measured number filled in.

    python proposal/build_proposal.py                       # uses results/pilot and models/sama_real
    python proposal/build_proposal.py --pilot results/pilot_SYNTHETIC --bundle models/x --allow-synthetic

Placeholders in the Markdown sources:
  {{P:path.to.key:fmt}}   value from <pilot>/pilot_summary.json
  {{T:path.to.key:fmt}}   value from <bundle>/sama_model.json "meta" (the public-data training run)
  {{V:key:fmt}}           SYNTHETIC verification numbers (technical report only)
  fmt: a Python format spec (".2f", "+.1f", "d"), or "pct" (x100, no decimals, with %), or "list".
A value that is missing renders as **[pending: P:path]** and is listed at the end of the build; the
script exits with status 2 if anything is pending in the proposal, so an unmeasured number cannot
slip into the upload. Inputs labelled SYNTHETIC are refused unless --allow-synthetic is given, and
then every page carries a "SYNTHETIC — NOT FOR SUBMISSION" watermark.

Also checked: portal character limits (summary 3,000; objectives and outcomes 1,500 each), that
the budget table's columns add up to its totals, and that no "[TO FILL: ...]" field (ethics reference,
date of joining, existing equipment) is left; any of these also gives exit status 2.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "proposal"
DOCS = [("SAMA_proposal.md", "SAMA_proposal"), ("SAMA_pilot_and_submission_plan.md", "SAMA_pilot_and_submission_plan"),
        ("SAMA_reviewer_stress_test.md", "SAMA_reviewer_stress_test"), ("CHANGES.md", "SAMA_proposal_changes"),
        ("../report/SAMA_technical_report_v2.md", "SAMA_technical_report_v2")]
CSS = """
@page { size: A4; margin: 2.2cm 2.2cm 2.4cm; }
html { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #14202B; }
h1.title { font-size: 17pt; margin: 0 0 4pt; } p.subtitle { font-size: 11pt; color: #46535F; margin: 0; }
p.author { color: #46535F; margin: 2pt 0 12pt; }
h1 { font-size: 14pt; margin: 16pt 0 6pt; border-bottom: 1px solid #C9D1DA; padding-bottom: 2pt; }
h2 { font-size: 12pt; margin: 12pt 0 4pt; } h3 { font-size: 11pt; margin: 8pt 0 3pt; }
p { margin: 0 0 6pt; text-align: justify; } li { margin: 0 0 3pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 4pt 0 10pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #B8C0C8; padding: 3pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 1.2pt solid #14202B; background: #F1F4F7; }
figure { margin: 8pt 0 10pt; page-break-inside: avoid; text-align: center; } figure img { max-width: 100%; }
figcaption { font-size: 9pt; text-align: left; color: #333; margin-top: 3pt; }
.note { border: 1px solid #D9A066; background: #FFF6E8; padding: 6pt 8pt; margin: 6pt 0 10pt; font-size: 9.5pt; }
.pending { background: #FFE08A; font-weight: bold; }
.pb { page-break-before: always; }
.wm { position: fixed; top: 40%; left: 5%; font-size: 46pt; color: rgba(200,0,0,0.13); transform: rotate(-30deg); z-index: 9; }
"""


def get(d, path):
    v = d
    for k in path.split("."):
        if isinstance(v, list):
            v = v[int(k)]
        elif isinstance(v, dict) and k in v:
            v = v[k]
        else:
            return None
    return v


def fmt(v, f):
    if v is None:
        return None
    try:
        if f == "pct":
            return f"{100 * float(v):.0f} %"
        if f == "list":
            return ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
        if f == "d":
            return f"{int(round(float(v)))}"
        if f:
            x = float(v)
            if x != x:          # nan
                return None
            return format(x, f)
        return str(v)
    except (TypeError, ValueError):
        return None


def resolve(md: str, sources: dict, pending: list) -> str:
    def sub(m):
        src, path, f = m.group(1), m.group(2), m.group(3) or ""
        v = fmt(get(sources.get(src) or {}, path), f)
        if v is None:
            pending.append(f"{src}:{path}")
            return f'<span class="pending">[pending: {src}:{path}]</span>'
        return v
    return re.sub(r"\{\{([PTV]):([A-Za-z0-9_.\-]+)(?::([^}]+))?\}\}", sub, md)


def check_limits(md: str) -> list:
    out = []
    for tag, lim in (("SUMMARY", 3000), ("OBJECTIVES", 1500), ("OUTCOME", 1500)):
        m = re.search(f"<!--{tag}-->\n(.*?)\n<!--/{tag}-->", md, re.S)
        if m:
            n = len(m.group(1))
            md = md.replace(f"@@{tag}_CHARS@@", f"{n:,}")
            out.append((tag, n, lim))
    return md, out


def check_budget(md: str) -> list:
    """Every column of the budget table adds up to its subtotal and grand total rows."""
    rows = [l for l in md.splitlines() if l.startswith("| ") and re.search(r"\| *\**[\d,]+\** *\|$", l)]
    num = lambda s: int(s.replace(",", "").replace("*", "").strip()) if re.search(r"\d", s) else 0  # noqa: E731
    items, eq, rec, grand = [], None, None, None
    for l in rows:
        cells = [c.strip() for c in l.strip("|").split("|")]
        if len(cells) != 6:
            continue
        vals = [num(c) for c in cells[2:]]
        if "Equipment total" in cells[0]:
            eq = vals
        elif "Recurring total" in cells[0]:
            rec = vals
        elif "Grand total" in cells[0]:
            grand = vals
        else:
            items.append((cells[0], vals))
    errs = []
    if not items or grand is None:
        return ["budget table not found"]
    for name, v in items:
        if sum(v[:3]) != v[3]:
            errs.append(f"row '{name}' years do not add up to its total")
    e = [sum(v[j] for n, v in items if n == "Equipment") for j in range(4)]
    r = [sum(v[j] for n, v in items if n != "Equipment") for j in range(4)]
    if eq and e != eq:
        errs.append(f"equipment total {eq} != sum {e}")
    if rec and r != rec:
        errs.append(f"recurring total {rec} != sum {r}")
    if [a + b for a, b in zip(e, r)] != grand:
        errs.append(f"grand total {grand} != {[a + b for a, b in zip(e, r)]}")
    return errs


def render(name, out_base, md, synthetic, outdir):
    from playwright.sync_api import sync_playwright
    outdir.mkdir(parents=True, exist_ok=True)
    resolved = outdir / f"{out_base}.md"
    resolved.write_text(md)
    html = subprocess.run(["pandoc", "-f", "markdown+pipe_tables+raw_html+implicit_figures+subscript", "-t", "html5", "-s",
                           "--metadata", "pagetitle=SAMA", "--resource-path", str(ROOT)],
                          input=md, capture_output=True, text=True, check=True).stdout
    html = html.replace("</head>", f"<style>{CSS}</style></head>")
    if synthetic:
        html = html.replace("<body>", '<body><div class="wm">SYNTHETIC — NOT FOR SUBMISSION</div>')
    html = re.sub(r'src="(?!file:|http)([^"]+)"', lambda m: f'src="file://{ROOT}/{m.group(1)}"', html)
    (outdir / f"{out_base}.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(f"file://{outdir}/{out_base}.html")
        pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(outdir / f"{out_base}.pdf"), format="A4", display_header_footer=True,
               margin={"top": "2.2cm", "bottom": "2.4cm", "left": "2.2cm", "right": "2.2cm"},
               header_template="<span></span>",
               footer_template="<div style='font-size:8pt;width:100%;text-align:center;color:#666'>SAMA · <span class='pageNumber'></span> of <span class='totalPages'></span></div>")
        b.close()
    docx_md = re.sub(r"<div class=\"pb\"></div>", "", md)
    docx_md = re.sub(r'<span class="pending">(.*?)</span>', r"**\1**", docx_md)
    subprocess.run(["pandoc", "-f", "markdown+pipe_tables+raw_html+implicit_figures+subscript", "-o", str(outdir / f"{out_base}.docx"),
                    "--resource-path", str(ROOT)], input=docx_md, text=True, check=True)
    pages = subprocess.run(["pdfinfo", str(outdir / f"{out_base}.pdf")], capture_output=True, text=True).stdout
    n = re.search(r"Pages:\s+(\d+)", pages)
    return int(n.group(1)) if n else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=str(ROOT / "results/pilot"))
    ap.add_argument("--bundle", default=str(ROOT / "models/sama_real"))
    ap.add_argument("--out", default=str(P / "build"))
    ap.add_argument("--verify", default=str(ROOT / "results/verification/verification_synthetic.json"),
                    help="SYNTHETIC end-to-end verification numbers for the technical report (V:)")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--only", help="build only this source file")
    a = ap.parse_args(argv)
    pilot_p, bundle_p = Path(a.pilot) / "pilot_summary.json", Path(a.bundle) / "sama_model.json"
    S = json.loads(pilot_p.read_text()) if pilot_p.exists() else {}
    T = json.loads(bundle_p.read_text())["meta"] if bundle_p.exists() else {}
    V = json.loads(Path(a.verify).read_text()) if Path(a.verify).exists() else {}
    synthetic = S.get("provenance") == "SYNTHETIC" or "SYNTHETIC" in str(T.get("provenance", "")) or \
        any("SYNTHETIC" in str(r.get("name", "")) for r in T.get("recordings", []))
    if synthetic and not a.allow_synthetic:
        sys.exit("refusing to build from SYNTHETIC results; pass --allow-synthetic for a watermarked test build")
    if T and any("SYNTH" in str(r.get("dataset", "")) + str(r.get("name", "")) for r in T.get("recordings", [])):
        synthetic = True
    fig6 = Path(a.pilot) / "figure6_POS.png"
    status, exit_code = [], 0
    for src, base in DOCS:
        if a.only and a.only != src:
            continue
        p = P / src
        if not p.exists():
            continue
        md = p.read_text()
        md = md.replace("@@FIG6@@", str(fig6.relative_to(ROOT)) if fig6.exists() and fig6.is_relative_to(ROOT) else
                        (str(fig6) if fig6.exists() else "proposal/figures/Fig6_pilot_TEMPLATE.png"))
        md, limits = check_limits(md)
        pending = []
        md = resolve(md, {"P": S, "T": T, "V": V}, pending)
        if not fig6.exists() and "Fig6_pilot_TEMPLATE" in md:
            pending.append("figure 6 (template image)")
        budget = check_budget(md) if src == "SAMA_proposal.md" else []
        manual = sorted(set(re.findall(r"\[(?:number|Status[^\]]*|List any[^\]]*|TO FILL[^\]]*)\]|_{4,}", md))) if src == "SAMA_proposal.md" else []
        pages = render(src, base, md, synthetic, Path(a.out))
        status.append({"file": base, "pages": pages, "pending": pending, "limits": limits, "budget_errors": budget, "manual_fields": manual})
        if src == "SAMA_proposal.md" and (pending or budget or manual or any(n > lim for _, n, lim in limits)):
            exit_code = 2
    for s in status:
        print(f"{s['file']}: {s['pages']} pages")
        for tag, n, lim in s["limits"]:
            print(f"   {tag}: {n} of {lim} characters{'  ** OVER LIMIT **' if n > lim else ''}")
        if s["budget_errors"]:
            print("   BUDGET: " + "; ".join(s["budget_errors"]))
        elif s["file"] == "SAMA_proposal":
            print("   budget table adds up")
        if s["pending"]:
            print(f"   {len(s['pending'])} PENDING: " + ", ".join(sorted(set(s["pending"]))))
        if s["manual_fields"]:
            print(f"   {len(s['manual_fields'])} FIELD(S) TO FILL BY HAND (build not ready to submit): " + " | ".join(s["manual_fields"]))
    if synthetic:
        print("SYNTHETIC inputs: every page is watermarked NOT FOR SUBMISSION")
    print(f"outputs in {a.out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
