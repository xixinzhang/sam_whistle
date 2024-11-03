import cv2
import numpy as np
import networkx as nx

def extract_contour_from_mask(binary_mask):
    # Ensure the mask is in the correct format for contour extraction (0-255 scale)
    mask_255 = (binary_mask * 255).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def simplify_contour(contour, epsilon=5):
    # Approximate contour with fewer points using the Douglas-Peucker algorithm
    # epsilon defines the maximum distance from the original contour
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    return simplified_contour

def contour_to_graph(contour):
    G = nx.Graph()
    
    # Add nodes for each simplified point in the contour
    for i, point in enumerate(contour):
        x, y = point[0]  # Each point is in the form [[x, y]]
        G.add_node(i, pos=(x, y))
        
        # Add edges between consecutive points
        if i > 0:
            G.add_edge(i-1, i)
    
    # Optionally, close the contour by connecting the last point to the first
    if len(contour) > 2:
        G.add_edge(len(contour) - 1, 0)
    
    return G

def graph_to_mask(graph, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    # Extract node positions
    positions = nx.get_node_attributes(graph, 'pos')
    
    # Get edges and draw lines between nodes on the mask
    for edge in graph.edges():
        pt1 = tuple(map(int, positions[edge[0]]))  # Convert node positions to integers
        pt2 = tuple(map(int, positions[edge[1]]))
        cv2.line(mask, pt1, pt2, 1, 1)  # Draw in binary format (1 for foreground)
    
    return mask

# # Assuming your binary mask is a 0,1 valued numpy array
# binary_mask = cv2.imread('/home/asher/Desktop/projects/sam_road/cityscale/20cities/region_0_gt.png', 0)  # Load binary mask (0, 255)
# binary_mask = binary_mask // 255  # Convert to binary (0, 1)

# # Extract contours from the binary mask
# contours = extract_contour_from_mask(binary_mask)

# # Process each contour
# for contour in contours:
#     print()
#     print(contour.shape)
#     simplified_contour = simplify_contour(contour, epsilon=5)  # Simplify the contour
#     print(simplified_contour.shape)
#     graph = contour_to_graph(simplified_contour)

#     # Restore the mask from the graph
#     restored_mask = graph_to_mask(graph, binary_mask.shape)

#     # Visualize the original and restored masks
#     cv2.imwrite("./outputs/Original.png", (binary_mask * 255).astype(np.uint8))  # Display in 255 scale for visibility
#     cv2.imwrite("./outputs/Restored.png", (restored_mask * 255).astype(np.uint8))  # Display in 255 scale for visibility
#     # cv2.waitKey(0)
#     cv2.destroyAllWindows()
import pickle

gt_graph = pickle.load(open(f"/home/asher/Desktop/projects/sam_road/cityscale/20cities/region_0_refine_gt_graph.p",'rb'))
print(len(gt_graph))

