import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom

# Shepp-Logan phantom and its svd
phantom = shepp_logan_phantom()
U, s, V = np.linalg.svd(phantom, full_matrices=False)

# Compute the variance of the singular values
variance = (s ** 2) / np.sum(s ** 2)    

# Plot it
plt.figure(figsize=(12,8))
plt.plot(variance, marker='.', color='purple')
plt.title('Variance of Singular Values')
plt.xlabel('Singular Value Index')
plt.ylabel('Variance')

#Set this part to zoom in on the singular values
#plt.xlim(0, 25)  # Zoom in on the first _ singular values
#plt.ylim(0, 0.4)  # Zoom in on the variance range of (_,_)
# Find the number of singular values needed to reach 85% of the variance
cumulative_variance = np.cumsum(variance)
num_singular_values = np.argmax(cumulative_variance >= 0.85) + 1

plt.text(0.5, 0.95, f"Number of singular values needed for 85% variance: {num_singular_values}", transform=plt.gca().transAxes, fontsize=12, ha='center')
plt.show()
