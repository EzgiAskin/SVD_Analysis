import numpy as np
import matplotlib.pyplot as plt

# Define a matrix
matrix = np.array([[1, 2, 2, 9, 9], [8, 9, 1, 1, 0], [0, 0, 4, 6, 0]])

# Perform SVD
U, S, V = np.linalg.svd(matrix)

# Take absolute values of matrices U and V -> for visual representation purposes
U_abs = np.abs(U)
V_abs = np.abs(V)

# Define the labels for x-axis and y-axis
Ax_labels = ['LOTR', 'Doctor Strange', 'Narnia', 'Shrek', 'Inside Out']
Ay_labels = ['Alice', 'Bob', 'Cody']
Ux_labels = ['Animated', 'Fantasy', 'Family']
Vy_labels = ['Animated', 'Fantasy', 'Family', 'Other', 'Other']
fig, axs = plt.subplots(1, 4, figsize=(16, 8))
fig.subplots_adjust(wspace=0.5)

# A 
axs[0].set_title("A")
axs[0].imshow(matrix, aspect='auto', cmap='gray')
axs[0].set_xticks(range(len(Ax_labels)))
axs[0].set_xticklabels(Ax_labels, rotation=45, ha='right')
axs[0].set_yticks(range(len(Ay_labels)))
axs[0].set_yticklabels(Ay_labels)

# U
axs[1].set_title("U")
axs[1].imshow(U_abs, aspect='auto', cmap='gray')
axs[1].set_xticks(range(len(Ux_labels)))
axs[1].set_xticklabels(Ux_labels, rotation=45, ha='right')
axs[1].set_yticks(range(len(Ay_labels)))
axs[1].set_yticklabels(Ay_labels)

# ∑
axs[2].set_title("∑")
axs[2].imshow(np.diag(S), aspect='equal', cmap='gray')
axs[2].set_xticks([])
axs[2].set_yticks([])

# V
axs[3].set_title("V^T")
axs[3].imshow(V_abs, aspect='auto', cmap='gray')
axs[3].set_xticks(range(len(Ax_labels)))
axs[3].set_xticklabels(Ax_labels, rotation=45, ha='right')
axs[3].set_yticks(range(len(Vy_labels)))
axs[3].set_yticklabels(Vy_labels)

plt.figure(num=1)
plt.gcf().canvas.manager.set_window_title("SVD Visualization")
plt.show()
