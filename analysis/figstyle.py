"""Shared figure style (colour-blind-safe stratum colours, provenance captions)."""
import matplotlib
import matplotlib.pyplot as plt

COL = {"lighter": "#D9A066", "middle": "#A0612B", "darker": "#4F2B12", "SAMA": "#0B6E6E", "POS": "#2F4BA0",
       "best": "#2F4BA0", "grey": "#9AA3AD"}


def setup():
    for f in ("Inter", "DejaVu Sans"):
        if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
                         "axes.grid": True, "grid.color": "#E3E6EA", "axes.axisbelow": True})
