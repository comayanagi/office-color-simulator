============================================================

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(
    page_title="オフィスカラーシミュレーター",
    page_icon="🏢",
    layout="wide"
)

# タイトル
st.title("🏢 オフィスカラーシミュレーター")
st.markdown("---")

# サイドバー
st.sidebar.header("⚙️ 設定")
st.sidebar.markdown("画像をアップロードして、オフィスの各要素の色を変更できます")

# 画像アップロード
uploaded_file = st.file_uploader(
    "オフィス画像をアップロード",
    type=["jpg", "jpeg", "png"],
    help="JPEG, PNG形式の画像をアップロードしてください"
)

if uploaded_file is not None:
    # 画像読み込み
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # RGB→BGR変換（OpenCV用）
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    # HSV変換
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # レイアウト
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 オリジナル")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎨 シミュレーション結果")

        # タブで要素を切り替え
        tab1, tab2, tab3, tab4 = st.tabs([
            "🪑 椅子", 
            "🚪 パーティション/壁", 
            "💡 照明", 
            "🎨 カスタム"
        ])

        result = hsv.copy()

        # 椅子の色変更
        with tab1:
            st.markdown("### 椅子の色を変更")
            chair_color = st.select_slider(
                "色相",
                options=[
                    ("赤", 0),
                    ("オレンジ", 15),
                    ("黄色", 30),
                    ("緑", 60),
                    ("シアン", 90),
                    ("青", 120),
                    ("紺", 110),
                    ("紫", 140),
                    ("ピンク", 160),
                    ("茶色", 10)
                ],
                format_func=lambda x: x[0]
            )

            change_chair = st.checkbox("椅子の色を変更する", value=False, key="chair")

            if change_chair:
                # ベージュ系椅子を検出
                lower_beige = np.array([10, 20, 80])
                upper_beige = np.array([30, 150, 220])
                mask_chair = cv2.inRange(result, lower_beige, upper_beige)

                result[:,:,0] = np.where(mask_chair > 0, chair_color[1], result[:,:,0])

        # パーティション/壁の色変更
        with tab2:
            st.markdown("### パーティション/壁の色を変更")
            wall_color = st.select_slider(
                "色相",
                options=[
                    ("グレー（暗）", 0),
                    ("ベージュ", 20),
                    ("青", 110),
                    ("紺", 115),
                    ("緑", 80),
                    ("ブラウン", 15)
                ],
                format_func=lambda x: x[0],
                key="wall_slider"
            )

            change_wall = st.checkbox("パーティション/壁の色を変更する", value=False, key="wall")

            if change_wall:
                # グレー系パーティションを検出
                lower_gray = np.array([0, 0, 50])
                upper_gray = np.array([180, 50, 180])
                mask_wall = cv2.inRange(result, lower_gray, upper_gray)

                result[:,:,0] = np.where(mask_wall > 0, wall_color[1], result[:,:,0])
                # 彩度も調整
                result[:,:,1] = np.where(mask_wall > 0, 60, result[:,:,1])

        # 照明の色変更
        with tab3:
            st.markdown("### 照明（ペンダントライト）の色を変更")
            light_color = st.select_slider(
                "色相",
                options=[
                    ("イエロー", 25),
                    ("オレンジ", 15),
                    ("青", 110),
                    ("緑", 70),
                    ("白（変更なし）", 30)
                ],
                format_func=lambda x: x[0],
                key="light_slider"
            )

            change_light = st.checkbox("照明の色を変更する", value=False, key="light")

            if change_light:
                # 黄色系照明を検出
                lower_yellow = np.array([20, 50, 100])
                upper_yellow = np.array([35, 255, 255])
                mask_light = cv2.inRange(result, lower_yellow, upper_yellow)

                result[:,:,0] = np.where(mask_light > 0, light_color[1], result[:,:,0])

        # カスタム設定
        with tab4:
            st.markdown("### 詳細設定（上級者向け）")

            custom_hue = st.slider("色相 (Hue)", 0, 179, 0)
            custom_sat = st.slider("彩度 (Saturation)", 0, 255, 100)

            hue_range = st.slider("検出範囲 (Hue)", 0, 179, (0, 50))
            sat_range = st.slider("検出範囲 (Saturation)", 0, 255, (0, 100))
            val_range = st.slider("検出範囲 (Value)", 0, 255, (50, 200))

            if st.button("カスタムカラー適用"):
                lower_custom = np.array([hue_range[0], sat_range[0], val_range[0]])
                upper_custom = np.array([hue_range[1], sat_range[1], val_range[1]])
                mask_custom = cv2.inRange(result, lower_custom, upper_custom)

                result[:,:,0] = np.where(mask_custom > 0, custom_hue, result[:,:,0])
                result[:,:,1] = np.where(mask_custom > 0, custom_sat, result[:,:,1])

        # 結果を表示
        result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, use_container_width=True)

        # ダウンロードボタン
        result_pil = Image.fromarray(result_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 シミュレーション画像をダウンロード",
            data=byte_im,
            file_name="office_simulation.png",
            mime="image/png"
        )

else:
    st.info("👆 左のサイドバーから画像をアップロードしてください")

    # 使い方ガイド
    with st.expander("📖 使い方ガイド"):
        st.markdown("""
        ### 使い方

        1. **画像アップロード**
           - サイドバーから会議室やオフィスの写真をアップロード

        2. **要素の選択**
           - 椅子、パーティション、照明などのタブを選択

        3. **色の変更**
           - スライダーで好きな色を選択
           - チェックボックスで変更を適用

        4. **ダウンロード**
           - シミュレーション結果を画像として保存

        ### ポイント
        - 複数の要素を同時に変更可能
        - リアルタイムでプレビュー表示
        - 元の画像は保持されます

        ### トラブルシューティング
        - 色が変わらない場合：検出範囲を調整してください
        - カスタムタブで詳細設定が可能です
        """)

# フッター
st.markdown("---")
st.markdown("### 🏢 オフィスカラーシミュレーター by Streamlit")


============================================================
