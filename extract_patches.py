import numpy as np
import os

# Load npz — file is in root folder
test = np.load(r'D:\Internship work\Project\ucdnet_app\test.npz')

test_img1 = test['img1']   # (53, 64, 64, 13)
test_img2 = test['img2']

# Create output folder
os.makedirs(r'D:\Internship work\Project\ucdnet_app\test_patches', exist_ok=True)

# Save all patches
for i in range(len(test_img1)):
    np.save(rf'D:\Internship work\Project\ucdnet_app\test_patches\img1_patch_{i}.npy', test_img1[i])
    np.save(rf'D:\Internship work\Project\ucdnet_app\test_patches\img2_patch_{i}.npy', test_img2[i])
    print(f'Patch {i} saved')

print(f"Done! {len(test_img1)} pairs saved in test_patches/")