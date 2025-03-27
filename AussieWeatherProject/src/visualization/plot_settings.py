import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns

# -------------------------------------------------------------------
# Style **
# -------------------------------------------------------------------
# print(plt.style.available)

#colors = cycler(color=plt.get_cmap("tab10").colors)  # ["b", "r", "g"]
#colors = ["#ff9999","#66b3ff","#99ff99","#ffcc99"]

mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["figure.titlesize"] = 25

mpl.rcParams["axes.facecolor"] = "white" # Tengelyek hátterének beállítása
plt.rcParams["axes.edgecolor"] = "#333F4B"  # Tengely színe
mpl.rcParams["axes.grid"] = False # Rácsvonalak bekapcsolása
# mpl.rcParams["axes.prop_cycle"] = colors
mpl.rcParams["axes.linewidth"] = 1
# plt.rcParams["axes.labelsize"] = 14  # Tengelyfeliratok mérete
# plt.rcParams["axes.labelweight"] = "bold"  # Tengelyfeliratok vastagsága

mpl.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linestyle"] = "--"  # Rácsvonalak stílusa
#plt.rcParams["grid.linewidth"] = 0.6  # Rácsvonalak vastagsága

mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
plt.rcParams["xtick.labelsize"] = 12  # X tengely címkéinek mérete
plt.rcParams["ytick.labelsize"] = 12  # Y tengely címkéinek mérete

plt.rcParams["font.family"] = "serif"  # Serif betűtípus használata
plt.rcParams["font.size"] = 12  # Alapértelmezett betűméret

plt.rcParams["legend.loc"] = "upper right"  # Legend helyzete
plt.rcParams["legend.frameon"] = False  # Legenda háttér kikapcsolása


# sns.set_style("whitegrid")  # Seaborn fehér rácsos stílus
# sns.set_palette("muted")  # Seaborn színpaletta