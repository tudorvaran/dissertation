class Recommender:
    def __init__(self, tree, scaler, index_list):
        self.tree = tree
        self.scaler = scaler
        self.index_list = index_list

    def flatten(self, lst):
        return [item for sublst in lst for item in sublst]

    def recommend(self, vectors, k, exclude_index=None):
        if self.scaler:
            vectors = self.scaler.transform(vectors)

        distances, leafs = self.tree.query(vectors, k=k + 1, return_distance=True)

        distances = self.flatten(distances)
        leafs = self.flatten(leafs)

        photo_ids = [self.index_list[leaf] for i, leaf in enumerate(leafs) if (exclude_index and self.index_list[leaf] != exclude_index) or not exclude_index]

        items = [dict(d=distances[i], i=photo_id) for i, photo_id in enumerate(photo_ids)]
        items.sort(key=lambda item: item['d'])

        return items[:k]
