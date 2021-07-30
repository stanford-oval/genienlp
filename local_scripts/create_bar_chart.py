import os
import sys

import flag
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

languages = ('sa', 'de', 'es', 'ir', 'fi', 'it', 'jp', 'pl', 'tr', 'cn')

lang2country = {
    'sa': 'saudi-arabia',
    'de': 'germany',
    'es': 'spain',
    'ir': 'iran',
    'fi': 'finland',
    'it': 'italy',
    'jp': 'japan',
    'pl': 'poland',
    'tr': 'turkey',
    'cn': 'china',
}

path_to_flags = sys.argv[1]


def get_flag(lang):
    path = os.path.join(path_to_flags, "{}.png".format(lang))
    im = plt.imread(path)
    return im


def offset_image(coord, name, ax):
    img = get_flag(name)
    im = OffsetImage(img, zoom=1)
    im.image.axes = ax
    ab = AnnotationBbox(im, (coord, 0), xybox=(27, -40), frameon=False, xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)


N = 10
x_labels = [flag.flag(lang.upper()) for lang in languages]
genie = (74.6, 77.1, 77.5, 74.2, 68.1, 69.0, 70.5, 64.3, 74.6, 65.3)
sota = (34.6, 52.3, 58.2, 57.8, 53.8, 56.1, 49.6, 59.6, 57.8, 42.8)

ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(25, 15))

rec1 = ax.bar(ind, sota, width, align="center", color='orange')
rec2 = ax.bar(ind + width, genie, width, align="center", color='dodgerblue')


ax.set_xticks(ind + width / 2)
ax.set_xticklabels(languages, fontsize=30)
ax.set_yticks(range(0, 110, 20))
ax.set_yticklabels([str(num) + '%' for num in range(0, 110, 20)], fontsize=60)
ax.tick_params(axis='x', which='major', pad=26)

ax.legend((rec2, rec1), ('Genie', 'SOTA'), fontsize=60)

ax.yaxis.grid(which='both', color='black', linestyle='-.')

for i, c in enumerate(languages):
    offset_image(i, c, ax)

plt.tight_layout()
plt.savefig('spl.pdf')
plt.show()
