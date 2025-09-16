import numpy as np

class KMeansClusterer:
    def __init__(self, data, num_clusters, max_iterations=100):
        self.data = data
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.centroids = None

    def _initialize_centroids(self, X):
        """Khởi tạo ngẫu nhiên các centroid từ dữ liệu"""
        n_samples = X.shape[0]
        if n_samples < self.num_clusters:
            raise ValueError(f"Số lượng mẫu ({n_samples}) phải lớn hơn hoặc bằng số cụm ({self.num_clusters}).")
        indices = np.random.choice(n_samples, self.num_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        """Gán mỗi điểm dữ liệu vào cụm gần nhất"""
        if X.shape[0] != len(self.data):
            raise ValueError(f"Số lượng điểm dữ liệu ({X.shape[0]}) không khớp với dữ liệu gốc ({len(self.data)}).")
        distances = np.zeros((X.shape[0], self.num_clusters))
        for k in range(self.num_clusters):
            distances[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Cập nhật centroid bằng cách tính trung bình các điểm trong mỗi cụm"""
        if len(labels) != X.shape[0]:
            raise ValueError(f"Số lượng nhãn ({len(labels)}) không khớp với số lượng điểm ({X.shape[0]}).")
        centroids = np.zeros((self.num_clusters, X.shape[1]))
        for k in range(self.num_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids

    def cluster(self):
        """Thực hiện thuật toán K-Means thủ công, trả về labels và centroids"""
        X = self.data.values

        # Kiểm tra dữ liệu đầu vào
        if X.shape[0] != len(self.data):
            raise ValueError(f"Số lượng hàng trong dữ liệu ({X.shape[0]}) không khớp với DataFrame gốc ({len(self.data)}).")

        # Khởi tạo centroids
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iterations):
            # Gán cụm
            labels = self._assign_clusters(X, self.centroids)
            
            # Cập nhật centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Kiểm tra hội tụ
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids

        return labels, self.centroids