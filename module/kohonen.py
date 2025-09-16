import numpy as np

class KohonenSOM:
    def __init__(self, data, num_clusters=3, learning_rate=0.5, max_iterations=100,
                 neighborhood_radius=None, init_method="random", manual_weights=None):
        self.data = data
        self.num_clusters = num_clusters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.neighborhood_radius = neighborhood_radius if neighborhood_radius is not None else num_clusters / 2
        self.init_method = init_method
        self.manual_weights = manual_weights
        self.weights = None
        self.labels = None
        # Các giá trị chuẩn hóa sẽ dùng lại
        self._min_vals = None
        self._ptp_vals = None

    def _normalize_data(self, X):
        # Chuẩn hóa dữ liệu về [0, 1] trên từng chiều
        self._min_vals = np.min(X, axis=0)
        self._ptp_vals = np.ptp(X, axis=0) + 1e-8
        return (X - self._min_vals) / self._ptp_vals

    def _normalize_manual_weights(self, weights):
        # Chuẩn hóa manual_weights giống như dữ liệu
        return (weights - self._min_vals) / self._ptp_vals

    def _denormalize_weights(self, weights):
        # Chuyển trọng số về giá trị gốc (không chuẩn hóa)
        return weights * self._ptp_vals + self._min_vals

    def _initialize_weights(self, X):
        n_features = X.shape[1]
        if self.init_method == "random":
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            self.weights = np.random.uniform(min_vals, max_vals, (self.num_clusters, n_features))
        elif self.init_method == "linear":
            indices = np.linspace(0, len(X) - 1, self.num_clusters).astype(int)
            self.weights = np.array([X[i] for i in indices])
        elif self.init_method == "manual":
            if self.manual_weights is None:
                raise ValueError("Bạn phải truyền vào manual_weights khi chọn khởi tạo manual.")
            arr = np.array(self.manual_weights, dtype=float)
            if arr.shape != (self.num_clusters, n_features):
                raise ValueError(f"manual_weights phải có shape ({self.num_clusters}, {n_features}), hiện tại là {arr.shape}")
            arr = self._normalize_manual_weights(arr)
            self.weights = arr
        else:
            raise ValueError("Phương thức khởi tạo không hợp lệ. Sử dụng 'random', 'linear' hoặc 'manual'.")

    def _euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def _find_bmu(self, point):
        dists = np.linalg.norm(self.weights - point, axis=1)
        return np.argmin(dists)

    def _update_weights(self, point, bmu_idx, iteration):
        current_lr = self.learning_rate * (1 - iteration / self.max_iterations)
        current_radius = self.neighborhood_radius * (1 - iteration / self.max_iterations)
        min_radius = 1e-6
        if current_radius < min_radius:
            current_radius = min_radius
        for j in range(self.num_clusters):
            dist_to_bmu = np.abs(j - bmu_idx)
            influence = np.exp(-dist_to_bmu ** 2 / (2 * (current_radius ** 2)))
            self.weights[j] += influence * current_lr * (point - self.weights[j])

    def cluster(self):
        # Bước 1: Chuẩn hóa dữ liệu
        X_norm = self._normalize_data(self.data)
        n_samples = X_norm.shape[0]
        # Bước 2: Khởi tạo trọng số
        self._initialize_weights(X_norm)
        # Bước 3: Huấn luyện SOM
        for iteration in range(self.max_iterations):
            indices = np.random.permutation(n_samples)
            for idx in indices:
                point = X_norm[idx]
                bmu_idx = self._find_bmu(point)
                self._update_weights(point, bmu_idx, iteration)
        # Bước 4: Gán nhãn cụm cho từng điểm dữ liệu
        self.labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            bmu_idx = self._find_bmu(X_norm[i])
            self.labels[i] = bmu_idx
        # Bước 5: Trả về trọng số gốc (không chuẩn hóa)
        weights_origin = self._denormalize_weights(self.weights)
        return self.labels, weights_origin

