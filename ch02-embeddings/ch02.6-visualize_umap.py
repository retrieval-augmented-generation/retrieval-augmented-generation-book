import umap
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from sentence_transformers import SentenceTransformer
from corpus import documents

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# Topic labels for color-coding (matches the 6 Acme Corp corpus categories)
topics = (
    ["Operations"] * 5 + ["Engineering"] * 5 + ["IT & Security"] * 5 +
    ["Product"] * 5 + ["Compliance"] * 5 + ["HR & Benefits"] * 5
)
color_map = {
    "Operations": "#1f77b4",
    "Engineering": "#ff7f0e",
    "IT & Security": "#2ca02c",
    "Product": "#d62728",
    "Compliance": "#9467bd",
    "HR & Benefits": "#8c564b",
}

# Reduce to 2D
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
coords = reducer.fit_transform(embeddings)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))

for topic in color_map:
    mask = np.array([t == topic for t in topics])
    points = coords[mask]
    ax.scatter(points[:, 0], points[:, 1],
               c=color_map[topic], label=topic, s=100, alpha=0.8, zorder=3)

    # Draw a convex hull around each cluster with slight padding
    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]
        centroid = points.mean(axis=0)
        padded = centroid + 1.15 * (hull_pts - centroid)
        poly = Polygon(padded, closed=True,
                       edgecolor=color_map[topic], facecolor=color_map[topic],
                       alpha=0.08, linewidth=1.5, zorder=1)
        ax.add_patch(poly)

# Annotate a few points for reference
for i in [0, 5, 10, 15, 20, 25]:
    ax.annotate(
        documents[i][:40] + "...",
        (coords[i, 0], coords[i, 1]),
        fontsize=7,
        alpha=0.8,
    )

ax.legend(loc="lower left")
ax.set_title("Acme Corp Document Embeddings Projected to 2D with UMAP")
ax.set_xlabel("UMAP dimension 1")
ax.set_ylabel("UMAP dimension 2")
fig.tight_layout()
fig.savefig("embedding_clusters.png", dpi=150)
plt.show()
