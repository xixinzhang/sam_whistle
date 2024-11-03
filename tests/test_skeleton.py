from skimage.morphology import medial_axis, skeletonize
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cv2

from shapely.geometry import LineString
from skimage import data
import sknw

binary_mask = cv2.imread('/home/asher/Desktop/projects/sam_road/cityscale/20cities/region_0_gt.png', 0)  # Load binary mask (0, 255)

# Medial axis skeletonization
# skeleton = medial_axis(binary_mask, )
skeleton = skeletonize(binary_mask, )

# # Visualize the skeleton
# plt.imshow(skeleton, cmap='gray')
# plt.savefig('outputs/skeleton.png')
# plt.close()
# plt.imshow(binary_mask, cmap='gray')
# plt.savefig('outputs/mask.png')


# # open and skeletonize
img = data.horse()
ske = skeletonize(~img).astype(np.uint8)
# img = binary_mask
# ske = skeleton

# build graph from skeleton
multi_edge = True
graph = sknw.build_sknw(ske, multi=multi_edge)


# draw image
plt.imshow(ske, cmap='gray')
# plt.imshow(img, cmap='gray')

def simplify_path(pts, tolerance=0.5):
    """Simplify the sequence of points between nodes using Shapely."""
    line = LineString(pts)
    simplified_line = line.simplify(tolerance)
    return np.array(simplified_line.coords)

tolerance = 0.5
# draw edges by pts
simplified_graph = nx.Graph()
pre_id = -1
cur_id = 0

for (s,e) in graph.edges():
    if multi_edge:
        for i in range(len(graph[s][e])):
            ps = graph[s][e][i]['pts']
            ps  = simplify_path(ps, tolerance=tolerance)

            for j, coord in enumerate(ps):
                simplified_graph.add_node(cur_id, pos=coord)
                if j > 0:
                    simplified_graph.add_edge(pre_id, cur_id)
                pre_id = cur_id
                cur_id += 1
            # print(nodes[s]['o'] , nodes[e]['o'])
            # print(ps)

            # plt.plot(ps[:,1], ps[:,0], 'green')
    else:
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')
    
# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])
print(len(nodes))
# plt.plot(ps[:,1], ps[:,0], 'r.')

print(len(simplified_graph.nodes()))
image = np.zeros(ske.shape + (3,), dtype=np.uint8) 
for (s, e) in simplified_graph.edges():
    pt1 = tuple(map(int, simplified_graph.nodes[s]['pos']))
    pt2 = tuple(map(int, simplified_graph.nodes[e]['pos']))
    cv2.line(image, pt1[::-1], pt2[::-1],  (0, 0, 255), 2)
    # plt.plot([pt1[1], pt2[1]], [pt1[0], pt2[0]], 'r')

cv2.imwrite('outputs/skeleton.png', image)
# # title and show
# plt.title('Build Graph')
# plt.show()
# plt.imshow(ske, cmap='gray')
# plt.show()