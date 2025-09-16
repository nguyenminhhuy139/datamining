import math
import numpy as np

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=10):
        self.criterion = criterion.lower()  # 'gini' hoặc 'entropy'
        self.max_depth = max_depth
        self.tree = None
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        """Xây dựng cây quyết định từ dữ liệu X và nhãn y."""
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(X.shape[1])]
        data = np.column_stack((X, y))
        self.tree = self._build_tree(data, depth=0)

    def _calculate_gini(self, y):
        """Tính Gini Index cho một tập nhãn."""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _calculate_entropy(self, y):
        """Tính Entropy cho một tập nhãn."""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(p * math.log2(p) for p in probabilities if p > 0)

    def _split_data(self, data, feature_idx):
        """Chia dữ liệu theo một thuộc tính (feature)."""
        unique_values = np.unique(data[:, feature_idx])
        splits = {value: data[data[:, feature_idx] == value] for value in unique_values}
        return splits

    def _information_gain(self, data, feature_idx, criterion):
        """Tính Information Gain (dựa trên Gini hoặc Entropy)."""
        total = len(data)
        y = data[:, -1]
        if criterion == 'gini':
            parent_score = self._calculate_gini(y)
        else:  # entropy
            parent_score = self._calculate_entropy(y)

        splits = self._split_data(data, feature_idx)
        child_score = 0
        for value_data in splits.values():
            if len(value_data) == 0:
                continue
            child_y = value_data[:, -1]
            weight = len(value_data) / total
            if criterion == 'gini':
                child_score += weight * self._calculate_gini(child_y)
            else:
                child_score += weight * self._calculate_entropy(child_y)

        return parent_score - child_score

    def _build_tree(self, data, depth):
        """Xây dựng cây quyết định đệ quy."""
        y = data[:, -1]
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(data[0]) == 1:
            return {'leaf': True, 'value': np.unique(y)[0]}

        best_feature = None
        max_gain = -float('inf')
        for feature_idx in range(data.shape[1] - 1):  # Không tính cột nhãn
            gain = self._information_gain(data, feature_idx, self.criterion)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature_idx

        if max_gain == 0:
            return {'leaf': True, 'value': np.unique(y)[0]}

        tree = {'leaf': False, 'feature': best_feature, 'children': {}}
        splits = self._split_data(data, best_feature)
        for value, value_data in splits.items():
            if len(value_data) == 0:
                tree['children'][value] = {'leaf': True, 'value': np.unique(y)[0]}
            else:
                tree['children'][value] = self._build_tree(value_data[:, [i for i in range(data.shape[1]) if i != best_feature]], depth + 1)
        return tree

    def _export_text_tree(self, node, prefix="", is_last=True, is_root=False):
        lines = []
        # 1) In node gốc
        if is_root:
            # chỉ root mới không có prefix con
            lines.append(f"{prefix}└── {self.feature_names[node['feature']]}")
        # 2) Nếu là lá thì in luôn giá trị
        if node.get('leaf', False):
            # (thường node lá chỉ in qua parent, nên hiếm khi vào đây)
            connector = "└──" if is_last else "├──"
            lines.append(f"{prefix}{connector} {node['value']}")
            return "\n".join(lines)

        # Lấy danh sách children để xác định last/others
        children = list(node['children'].items())
        for idx, (branch_value, child) in enumerate(children):
            last_child = (idx == len(children) - 1)
            connector = "└──" if last_child else "├──"

            # prefix cho dòng này
            child_prefix = prefix + ("    " if is_last else "│   ")

            if child.get('leaf', False):
                # in thẳng giá trị nhánh: class
                lines.append(f"{child_prefix}{connector} {branch_value} : {child['value']}")
            else:
                # in nhánh rồi đệ quy
                lines.append(f"{child_prefix}{connector} {branch_value}")
                # prefix tiếp cho con của con
                sub_prefix = child_prefix + ("    " if last_child else "│   ")
                lines.append(self._export_text_tree(child,
                                                    prefix=sub_prefix,
                                                    is_last=last_child,
                                                    is_root=False))
        return "\n".join(lines)

    def export_text_tree_manual(self, feature_names=None):
        """Khởi tạo và gọi _export_text_tree với flag is_root=True"""
        if feature_names is not None:
            self.feature_names = feature_names
        return self._export_text_tree(self.tree, prefix="", is_last=True, is_root=True)
