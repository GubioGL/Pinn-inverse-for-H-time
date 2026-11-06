"""Figure style for the paper (APS two-column journals: Phys. Rev. A/Research/Applied).

    import estilo
    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(estilo.COL, 2.2))

* widths: COL = 3.375 in (one column), DUPLA = 7.0 in (two columns);
* fonts: LaTeX Computer Modern when a LaTeX installation is found (same font as revtex),
  otherwise matplotlib's Computer Modern math with a serif text font;
* colours: Okabe-Ito palette (colour-blind safe); every series also has its own marker or
  line style, so the figures survive greyscale printing;
* vector output (PDF) through paths.save_figure.
"""
import shutil
import matplotlib as mpl

COL = 3.375      # in
DUPLA = 7.0      # in

# Okabe & Ito (2008) colour-blind safe palette
OI = {"preto": "#000000", "laranja": "#E69F00", "azul_claro": "#56B4E9", "verde": "#009E73",
      "amarelo": "#F0E442", "azul": "#0072B2", "vermelho": "#D55E00", "rosa": "#CC79A7",
      "cinza": "#7F7F7F"}

# one colour per method family, used in every figure
FAMILIA = {
    "classico": OI["cinza"],
    "original": OI["vermelho"],
    "lbfgs": OI["laranja"],
    "bfgs": OI["rosa"],
    "ssbfgs": "#8C6A00",          # darker than the Okabe-Ito yellow, readable on white
    "lm": OI["azul"],
    "lm_causal": OI["azul_claro"],
}


def aplicar(usetex=None):
    if usetex is None:
        usetex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    rc = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=[OI["azul"], OI["vermelho"], OI["verde"], OI["laranja"],
                                             OI["rosa"], OI["azul_claro"], OI["preto"]]),
    }
    if usetex:
        rc.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                   "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}"})
    else:
        rc.update({"text.usetex": False, "font.family": "serif",
                   "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
                   "mathtext.fontset": "cm"})
    mpl.rcParams.update(rc)
    return usetex


def rotulo_painel(ax, letra, x=-0.18, y=1.0):
    """Panel label (a), (b), ... in the upper-left corner, outside the axes."""
    ax.text(x, y, f"({letra})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold" if not mpl.rcParams["text.usetex"] else "normal")
