import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

all_2_class_bboxes = np.load("all_bboxes_random_flip.npy")
all_x_means = all_2_class_bboxes[:,0] / 2 + all_2_class_bboxes[:,2] / 2
all_y_means = all_2_class_bboxes[:,1] / 2 + all_2_class_bboxes[:,3] / 2
all_size_ratios = (all_2_class_bboxes[:,2] - all_2_class_bboxes[:,0]) * (all_2_class_bboxes[:,3] - all_2_class_bboxes[:,1]) / (640*480)
all_aspect_ratios = (all_2_class_bboxes[:,2] - all_2_class_bboxes[:,0]) / (all_2_class_bboxes[:,3] - all_2_class_bboxes[:,1]) # x / y
all_colors = np.vstack([np.array([1,0,0]) if i == 1 else np.array([0,0,1]) for i in all_2_class_bboxes[:,-1]])

plt.figure(figsize=(24,18))
plt.xlim(0,640)
plt.ylim(0,480)
plt.title('bbox center positions plot')
plt.scatter(all_x_means, all_y_means, c=all_colors, s=2)
plt.show()

fig = plt.figure(figsize=(30,20),dpi=80)
ax = Axes3D(fig)
ax.set_title('bbox_size / figure_size ratios 3d plot')
ax.scatter(all_x_means, all_y_means, all_size_ratios, s=1, c=all_colors)
plt.show()

fig = plt.figure(figsize=(30,20),dpi=80)
ax = Axes3D(fig)
ax.set_title('bbox aspect ratios 3d plot')
ax.scatter(all_x_means, all_y_means, all_aspect_ratios, s=1, c=all_colors)
plt.show()

fig = plt.figure(figsize=(30,20),dpi=80)
ax = Axes3D(fig)
ax.set_title('bbox aspect log ratios 3d plot')
ax.scatter(all_x_means, all_y_means, np.log(all_aspect_ratios), s=1, c=all_colors)
plt.show()