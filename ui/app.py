# ui/app.py – Giao diện nhiều trang (Trang chọn mô hình + Trang thao tác)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from module import apriori_custom
from module.naive_bayes_custom import NaiveBayesClassifier
from module.rough_set_custom import RoughSet
from module.k_means import KMeansClusterer
from module.decision_tree import DecisionTree
from module.kohonen import KohonenSOM

st.set_page_config(page_title="Khai phá dữ liệu", layout="wide")

# --- Khởi tạo session state ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Trang chọn mô hình ---
def show_home():
    st.markdown("<h1 style='text-align: center;'>📚 Chọn mô hình khai phá dữ liệu</h1>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if st.button("📊 Apriori (Luật kết hợp)", use_container_width=True):
            st.session_state.page = 'apriori'

    with col2:
        if st.button("🧠 Naive Bayes (Phân lớp)", use_container_width=True):
            st.session_state.page = 'bayes'
    
    with col3:
        if st.button("📘 Tập thô (Rough Set)", use_container_width=True):
            st.session_state.page = 'rough'

    with col4:
        if st.button("📚 K-Means (Phân cụm)", use_container_width=True):
            st.session_state.page = 'kmeans'
    with col5:
        if st.button("🌳 Cây quyết định", use_container_width=True):
            st.session_state.page = 'decision_tree'
    with col6:
        if st.button("🗺️ Kohonen SOM (Phân cụm)", use_container_width=True):
            st.session_state.page = 'kohonen'

# --- Trang Apriori ---
def show_apriori():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("📊 Luật kết hợp - Apriori hoặc Không tăng cường")
    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="apriori_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        event_col = st.selectbox("🔹 Cột giao dịch:", df.columns)
        item_col = st.selectbox("🔸 Cột mặt hàng:", df.columns)

        st.radio("⚙️ Chọn thuật toán:", options=["Apriori", "Không tăng cường"], key="algo")

        min_sup = st.number_input("📏 Min Support (0–1):", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        min_conf = st.number_input("📐 Min Confidence (0–1):", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

        if st.button("🚀 Chạy luật kết hợp"):
            df_temp = df[[event_col, item_col]].dropna()
            df_temp.columns = ['Invoice', 'Item']
            transactions = apriori_custom.create_transactions(df_temp)

            freq_items = apriori_custom.find_frequent_itemsets(transactions, min_sup)
            rules = apriori_custom.generate_rules(freq_items, transactions, min_conf)

            st.subheader("✅ Tập phổ biến:")
            if not freq_items.empty:
                freq_items['itemsets'] = freq_items['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))
                st.dataframe(freq_items)
                csv_freq = freq_items.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải tập phổ biến", csv_freq, file_name="frequent_itemsets.csv", mime="text/csv")

                # 📊 Biểu đồ bar Support
                st.subheader("📊 Biểu đồ Support:")
                fig, ax = plt.subplots()
                top_freq = freq_items.sort_values('support', ascending=False).head(15)
                ax.barh(top_freq['itemsets'], top_freq['support'])
                ax.set_xlabel('Support')
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Không có tập phổ biến nào.")

            st.subheader("📐 Luật kết hợp:")
            if not rules.empty:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])

                csv_rules = rules.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải luật kết hợp", csv_rules, file_name="association_rules.csv", mime="text/csv")

                # 📈 Biểu đồ scatter Support vs Confidence
                st.subheader("📈 Biểu đồ Support vs Confidence:")
                fig2, ax2 = plt.subplots()
                ax2.scatter(rules['support'], rules['confidence'], alpha=0.6)
                ax2.set_xlabel('Support')
                ax2.set_ylabel('Confidence')
                ax2.set_title('Scatter plot: Support vs Confidence')
                st.pyplot(fig2)
            else:
                st.warning("⚠️ Không tìm thấy luật nào.")

# --- Trang Naive Bayes ---
def show_bayes():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("🧠 Naive Bayes - Phân lớp")
    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="bayes_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        target_col = st.selectbox("🎯 Cột mục tiêu:", df.columns)
        input_values = {}
        feature_cols = [col for col in df.columns if col != target_col]

        for col in feature_cols:
            input_values[col] = st.selectbox(f"🔹 {col}", df[col].unique())

        if st.button("🚀 Dự đoán"):
            clf = NaiveBayesClassifier()
            clf.fit(df, target_col)
            predicted_class, log_scores = clf.predict(input_values)

            st.success(f"✅ Dự đoán: `{predicted_class}`")
            st.json({k: round(v, 4) for k, v in log_scores.items()})

# ---Trang Rough Set---

def show_rough_set():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("📘 Tập thô – Rough Set")

    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="rough_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        decision_col = st.selectbox("🎯 Chọn cột quyết định (Decision Attribute):", df.columns)
        condition_cols = st.multiselect("🔹 Chọn các cột điều kiện (Condition Attributes):", [col for col in df.columns if col != decision_col])
        func = st.radio("⚙️ Chọn chức năng:", ["Xấp xỉ (Lower/Upper)", "Phụ thuộc thuộc tính", "Tìm reduct", "Sinh luật chính xác 100%"])

        if st.button("🚀 Thực hiện"):
            rs = RoughSet(df, condition_cols, decision_col)

            if func == "Xấp xỉ (Lower/Upper)":
                val = st.selectbox("🧪 Chọn giá trị quyết định:", df[decision_col].unique())
                lower = rs.lower_approx(val)
                upper = rs.upper_approx(val)

                st.write(f"📥 Lower approximation của `{val}` ({len(lower)} dòng):", sorted(lower))
                st.write(f"📤 Upper approximation của `{val}` ({len(upper)} dòng):", sorted(upper))

            elif func == "Phụ thuộc thuộc tính":
                degree = rs.dependency_degree()
                st.success(f"📊 Mức độ phụ thuộc: `{round(degree, 4)}`")

            elif func == "Tìm reduct":
                reduct = rs.find_reduct()
                st.success(f"🔍 Reduct tìm được: `{', '.join(reduct)}`")

            elif func == "Sinh luật chính xác 100%":
                rules = rs.generate_rules()
                if rules:
                    st.subheader(f"📜 {len(rules)} luật chính xác 100%:")
                    for i, r in enumerate(rules, 1):
                        cond = ' ∧ '.join([f"{k}={v}" for k, v in r['conditions'].items()])
                        st.write(f"**Luật {i}:** Nếu {cond} → {decision_col} = {r['decision']}")
                    # Cho phép tải
                    rule_df = pd.DataFrame([{
                        **r['conditions'], '=>': f"{decision_col}={r['decision']}"
                    } for r in rules])
                    csv = rule_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Tải các luật về CSV", csv, file_name="rough_rules.csv", mime='text/csv')
                else:
                    st.warning("⚠️ Không sinh được luật nào.")

# ---Trang K-mean---
def show_kmeans():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("📚 K-Means - Phân cụm")
    
    st.markdown("**Hướng dẫn**: Tải file CSV chứa các cột số để thực hiện phân cụm. Đảm bảo dữ liệu không chứa giá trị thiếu.")

    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="kmeans_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if df.isna().any().any():
            st.error("⚠️ Dữ liệu chứa giá trị thiếu. Vui lòng làm sạch dữ liệu trước khi chạy K-Means.")
            return
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) < 2:
            st.error("⚠️ Dữ liệu phải có ít nhất 2 cột số để thực hiện phân cụm.")
            return
        
        # Định dạng hiển thị: chỉ giữ 2 chữ số thập phân
        st.write("**Dữ liệu đầu vào:**")
        styled_df = df.style.format({col: "{:.2f}" for col in numeric_cols})
        st.dataframe(styled_df)

        num_clusters = st.number_input("🔢 Số lượng cụm (k):", min_value=2, max_value=len(df)-1, value=2, step=1)

        if st.button("🚀 Chạy K-Means"):
            if num_clusters >= len(df):
                st.error("⚠️ Số cụm phải nhỏ hơn số lượng mẫu trong dữ liệu.")
                return

            try:
                with st.spinner("Đang thực hiện phân cụm..."):
                    clusterer = KMeansClusterer(df[numeric_cols], num_clusters, max_iterations=100)
                    labels, centroids = clusterer.cluster()
                    
                    if len(labels) != len(df):
                        raise ValueError(f"Độ dài nhãn ({len(labels)}) không khớp với số hàng dữ liệu ({len(df)}).")

                    df['Cluster'] = labels

                st.markdown("### Kết quả thuật toán K-means")

                st.markdown("**Trọng tâm:**")
                for i, centroid in enumerate(centroids):
                    centroid_rounded = [round(x, 2) for x in centroid]
                    st.markdown(f"Trọng tâm {i + 1}: [{', '.join(map(str, centroid_rounded))}]")

                for cluster_id in range(num_clusters):
                    cluster_data = df[labels == cluster_id][numeric_cols].reset_index(drop=True)
                    st.markdown(f"**Cum {cluster_id + 1} (kích thước: {len(cluster_data)}):**")
                    # Định dạng hiển thị cho bảng cụm
                    styled_cluster = cluster_data.style.format({col: "{:.1f}" for col in numeric_cols})
                    st.table(styled_cluster)

            except Exception as e:
                st.error(f"⚠️ Lỗi khi chạy K-Means: {str(e)}")
                return

# ---Trang Decision Tree---          
def show_decision_tree():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("🌳 Cây quyết định")

    # Tải file CSV
    uploaded_file = st.file_uploader("📂 Tải file CSV", type=["csv"], key="decision_tree_upload")

    if uploaded_file:
        # Đọc dữ liệu
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        # Chọn cột mục tiêu và các cột đặc trưng
        target_column = st.selectbox("🎯 Chọn cột mục tiêu:", df.columns)
        feature_columns = [col for col in df.columns if col != target_column]

        # Lấy dữ liệu X (đặc trưng) và y (nhãn)
        X = df[feature_columns].values
        y = df[target_column].values

        # Chọn tiêu chí phân chia
        criterion = st.radio("⚙️ Chọn tiêu chí phân chia:", ["Gini", "Entropy"])

        # Nút để bắt đầu xây dựng cây quyết định
        if st.button("🚀 Hiển thị cây quyết định"):
            # Sử dụng class DecisionTree
            tree_model = DecisionTree(criterion=criterion.lower())
            tree_model.fit(X, y)

            # Chuyển đổi cây quyết định thành dạng văn bản (thủ công)
            tree_text = tree_model.export_text_tree_manual(feature_names=feature_columns)

            # Hiển thị cây quyết định dạng văn bản
            st.subheader("🔍 Cây quyết định dạng văn bản:")
            st.code(tree_text, language="text")

#---Trang kohonen---
def show_kohonen():
    st.button("⬅️ Quay lại menu", on_click=lambda: st.session_state.update(page='home'))
    st.header("🗺️ Kohonen SOM - Phân cụm")

    st.markdown("**Hướng dẫn**: Tải file CSV chứa các cột số để thực hiện phân cụm. Đảm bảo dữ liệu không chứa giá trị thiếu.")

    uploaded_file = st.file_uploader("Tải lên tệp CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if df.isna().any().any():
            st.error("⚠️ Dữ liệu chứa giá trị thiếu. Vui lòng làm sạch dữ liệu trước khi chạy Kohonen SOM.")
            return

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) < 2:
            st.error("⚠️ Dữ liệu phải có ít nhất 2 cột số để thực hiện phân cụm.")
            return

        df[numeric_cols] = df[numeric_cols].round(2)

        st.subheader("📋 Xem trước dữ liệu")
        st.dataframe(df)

        selected_columns = st.multiselect(
            "Chọn các cột để phân cụm (Số chiều):",
            options=list(numeric_cols),
            default=list(numeric_cols[:3])
        )

        if not selected_columns:
            st.warning("⚠️ Vui lòng chọn ít nhất một cột để phân cụm.")
            return

        # Các thông số mô hình
        so_cum = st.number_input("Số Cụm", min_value=1, value=3, step=1)
        so_lap = st.number_input("Số Lần Lặp", min_value=1, value=5, step=1)
        ban_kinh_vong_lan_can = st.number_input("Bán Kính Vùng Lân Cận", min_value=0.0, value=0.0, step=0.1)
        toc_do_hoc = st.number_input("Tốc Độ Học", min_value=0.01, max_value=1.0, value=0.4, step=0.01)
        phuong_thuc_khoi_tao = st.selectbox("Phương thức khởi tạo trọng số:", options=["random", "manual"])

        manual_weights = None
        if phuong_thuc_khoi_tao == "manual":
            st.markdown("**Nhập trọng số thủ công cho từng cụm (dạng: x, y, z):**")

            num_clusters = so_cum
            num_dimensions = len(selected_columns)

            manual_weights = []
            for cluster_idx in range(num_clusters):
                weight_str = st.text_input(
                    f"Trọng số cho cụm {cluster_idx+1} (dạng: x, y, z):", 
                    key=f"manual_weight_line_{cluster_idx}", 
                    value=", ".join(["0.0"]*num_dimensions)
                )
                try:
                    cluster_weights = [float(x.strip()) for x in weight_str.split(",")]
                    if len(cluster_weights) != num_dimensions:
                        st.error(f"Số chiều nhập vào cho cụm {cluster_idx+1} phải đúng bằng {num_dimensions}.")
                        return
                    manual_weights.append(cluster_weights)
                except ValueError:
                    st.error(f"Trọng số cho cụm {cluster_idx+1} phải là các số, ngăn cách bằng dấu phẩy.")
                    return

        # Nút chạy thuật toán
        if st.button("Chạy Thuật Toán"):
            try:
                with st.spinner("Đang thực hiện phân cụm..."):
                    clusterer = KohonenSOM(
                        df[selected_columns].values,
                        num_clusters=so_cum,
                        learning_rate=toc_do_hoc,
                        max_iterations=so_lap,
                        neighborhood_radius=ban_kinh_vong_lan_can,
                        init_method=phuong_thuc_khoi_tao,
                        manual_weights=manual_weights
                    )
                    labels, weights = clusterer.cluster()

                    df['Cluster'] = labels

                st.markdown("### 🔍 Kết quả phân cụm Kohonen SOM")
                # Trọng tâm
                st.markdown("**🎯 Trọng tâm (node SOM):**")
                for idx, node_weights in enumerate(weights):
                    st.markdown(f"Node {idx + 1}: [{', '.join(map(lambda x: f'{x:.2f}', node_weights))}]")

                # Dữ liệu cụm
                cluster_ids = sorted(set(labels))
                for cluster_id in cluster_ids:
                    columns_to_show = ["Tranh"] + selected_columns if "Tranh" in df.columns else selected_columns
                    cluster_data = df[df['Cluster'] == cluster_id][columns_to_show]
                    if "Tranh" in cluster_data.columns:
                        cluster_data = cluster_data.set_index("Tranh")
                    st.markdown(f"**Cụm {cluster_id + 1} (số lượng: {len(cluster_data)}):**")
                    st.dataframe(cluster_data)

                if len(selected_columns) >= 2:
                    x_col = selected_columns[0]
                    y_col = selected_columns[1]
                    fig, ax = plt.subplots(figsize=(6, 6))
                    scatter = ax.scatter(
                        df[x_col], df[y_col],
                        c=df['Cluster'],
                        cmap='viridis',
                        s=100
                    )
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title("Phân Bố Các Cụm")
                    plt.colorbar(scatter, ax=ax)
                    st.markdown("**Biểu Đồ Phân Cụm:**")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Lỗi khi chạy Kohonen SOM: {str(e)}")

# --- Hiển thị trang tương ứng ---
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'apriori':
    show_apriori()
elif st.session_state.page == 'bayes':
    show_bayes()
elif st.session_state.page == 'rough':
    show_rough_set()
elif st.session_state.page == 'kmeans':
    show_kmeans()
elif st.session_state.page == 'decision_tree':
    show_decision_tree()
elif st.session_state.page == 'kohonen':
    show_kohonen()