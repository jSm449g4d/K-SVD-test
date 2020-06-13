import cv2
from matplotlib import pyplot as plt
import numpy as np
from spmimage.decomposition import KSVD
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

im =cv2.imread("./food_jambalaya.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)#cv2.COLOR_BGR2RGB
im = cv2.resize(im, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)/256.0
im=np.clip(im, 0.001, 0.999)
plt.title("origin");plt.imshow(im,vmin=0.0, vmax=1.0);plt.show()

ksvd = KSVD(n_components = 12, transform_n_nonzero_coefs = None ,n_jobs=12)
ksvd.fit(im.reshape(-1,64).astype(np.float64))

D = ksvd.components_.reshape(-1, 8,8).astype(np.float64)
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(D[i],vmin=0.0, vmax=1.0)
    plt.axis('off')
plt.show()

X = ksvd.transform(im.reshape(-1,64).astype(np.float64))
plt.title("X")
plt.imshow(X,vmin=0.0, vmax=1.0)
plt.show()

plt.title("Re")
_y=np.dot(X, ksvd.components_)
_y = _y.reshape(128, 128).astype(np.float64)
plt.imshow(_y,vmin=0.0, vmax=1.0)
plt.show()

print(compare_psnr(im, _y))
print(compare_ssim(im, _y))

