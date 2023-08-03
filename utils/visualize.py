import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def embedding_visualize(embs, user_num, item_num, title='embedding'):
    embs = embs.cpu().numpy()
    labels = np.array([0] * user_num + [1] * item_num)
    embs = TSNE(n_components=2).fit_transform(embs)

    plt.figure(figsize=(8, 8))
    plt.title(title)

    plt.scatter(embs[:, 0], embs[:, 1], c=labels, cmap='tab10')
    plt.show()
