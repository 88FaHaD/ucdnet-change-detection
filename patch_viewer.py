import numpy as np
import matplotlib.pyplot as plt
import os

def to_rgb(arr):
    rgb = arr[:, :, [3, 2, 1]].astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-7)
    return rgb

older_path = r'D:\Internship work\Project\ucdnet_app\test_patches\Older_date'
later_path = r'D:\Internship work\Project\ucdnet_app\test_patches\Later_date'

NUM_PATCHES = 53
current     = [0]  # mutable for closure

def show_patch(idx):
    img1 = np.load(os.path.join(older_path, f'img1_patch_{idx}.npy'))
    img2 = np.load(os.path.join(later_path, f'img2_patch_{idx}.npy'))

    axes[0].clear()
    axes[1].clear()

    axes[0].imshow(to_rgb(img1))
    axes[0].set_title(f'Patch {idx} — Older Date (T1)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(to_rgb(img2))
    axes[1].set_title(f'Patch {idx} — Later Date (T2)', fontsize=12)
    axes[1].axis('off')

    fig.suptitle(f'Patch {idx} / {NUM_PATCHES - 1} — Same Location Different Time',
                 fontsize=13, fontweight='bold')
    fig.canvas.draw()

def on_next(event):
    if current[0] < NUM_PATCHES - 1:
        current[0] += 1
        show_patch(current[0])

def on_prev(event):
    if current[0] > 0:
        current[0] -= 1
        show_patch(current[0])

# Setup figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.15)

# Buttons
from matplotlib.widgets import Button
ax_prev = plt.axes([0.3, 0.03, 0.15, 0.06])
ax_next = plt.axes([0.55, 0.03, 0.15, 0.06])
btn_prev = Button(ax_prev, '⬅ Previous')
btn_next = Button(ax_next, 'Next ➡')
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)

# Show first patch
show_patch(0)
plt.show()