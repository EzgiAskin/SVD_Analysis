import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom

# Shepp-Logan phantom and its svd
phantom = shepp_logan_phantom()
U, s, V = np.linalg.svd(phantom, full_matrices=False)

# Amount of singular values to keep
n = 2 
# Diagonal matrix with the first n singular values
S = np.diag(s[:n])

# U ∑ V with truncation of n 
recon_phantom = U[:, :n] @ S @ V[:n, :]

# Plot both the original phantom and reconstructed phantom
fig, phantoms = plt.subplots(1, 2)

# To remove the gray scale, remove the parameter cmap
phantoms[0].imshow(phantom, cmap='gray')
phantoms[0].set_title('Original Phantom')
phantoms[1].imshow(recon_phantom, cmap='gray')
phantoms[1].set_title(f'Reconstructed Phantom (n = {n})')

for a in phantoms:
    a.axis('off')
plt.show()
