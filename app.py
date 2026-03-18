import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import zipfile
import io

# --- 页面配置 ---
st.set_page_config(page_title="表情包全自动处理工具", layout="wide")
st.title("🎨 表情包处理助手 (去字+抠图+灵活下载)")

# --- 缓存 OCR 模型 ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

# --- 侧边栏设置 ---
st.sidebar.header("🛠️ 参数设置")
m = st.sidebar.number_input("行数 (m)", min_value=1, value=4)
n = st.sidebar.number_input("列数 (n)", min_value=1, value=3)
remove_text = st.sidebar.checkbox("去除中文文本", value=True)
threshold = st.sidebar.slider("抠图灵敏度 (背景去除)", 0, 100, 30)

st.sidebar.markdown("---")
download_mode = st.sidebar.radio("📥 下载方式选择", ["ZIP 压缩包 (推荐)", "手动单张批量下载"])

# --- 主界面逻辑 ---
uploaded_file = st.file_uploader("上传表情包图片 (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 加载图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="原图预览", use_container_width=True)

    if st.button("🚀 开始处理"):
        reader = load_ocr() if remove_text else None
        h, w, _ = img_bgr.shape
        tile_h, tile_w = h // m, w // n
        
        # 用于存储处理后的 PIL 图片对象及其文件名
        processed_images = []
        
        progress_bar = st.progress(0, text="正在处理图片...")
        total_steps = m * n
        step = 0

        for r in range(m):
            for c in range(n):
                step += 1
                # 1. 切割
                y_s, y_e = r*tile_h, (r+1)*tile_h
                x_s, x_e = c*tile_w, (c+1)*tile_w
                tile = img_bgr[y_s:y_e, x_s:x_e].copy()

                # 2. 去字
                if remove_text:
                    results = reader.readtext(tile)
                    mask = np.zeros(tile.shape[:2], dtype=np.uint8)
                    for (bbox, text, prob) in results:
                        if any('\u4e00' <= char <= '\u9fff' for char in text):
                            pts = np.array(bbox, np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                    if np.any(mask):
                        tile = cv2.inpaint(tile, mask, 3, cv2.INPAINT_NS)

                # 3. 抠图转透明
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(tile_rgb).convert("RGBA")
                data = pil_img.getdata()
                
                # 采样左上角像素作为背景参考
                bg_ref = data[0]
                new_data = []
                for item in data:
                    # 计算颜色差异 (欧氏距离简化版)
                    diff = sum(abs(item[i] - bg_ref[i]) for i in range(3))
                    new_data.append((0, 0, 0, 0) if diff < threshold else item)
                
                pil_img.putdata(new_data)
                
                # 保存到列表
                img_name = f"sticker_{r+1}_{c+1}.png"
                processed_images.append((pil_img, img_name))
                
                progress_bar.progress(step / total_steps)

        st.success("✨ 处理完成！")

        # --- 导出逻辑 ---
        if download_mode == "ZIP 压缩包 (推荐)":
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for pil_img, name in processed_images:
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG')
                    zip_file.writestr(name, img_byte_arr.getvalue())
            
            st.download_button(
                label="📥 点击下载所有图片 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="processed_stickers.zip",
                mime="application/zip",
                use_container_width=True
            )

        else:
            st.write("### 📸 单张图片预览与下载")
            st.info("您可以直接点击下方按钮下载，或右键图片“另存为”。")
            
            # 使用网格布局展示预览
            cols = st.columns(3) # 每行显示 3 个
            for idx, (pil_img, name) in enumerate(processed_images):
                with cols[idx % 3]:
                    st.image(pil_img, caption=name, use_container_width=True)
                    
                    # 准备单张下载的数据流
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    st.download_button(
                        label=f"保存 {name}",
                        data=buf.getvalue(),
                        file_name=name,
                        mime="image/png",
                        key=f"btn_{idx}"
                    )

# --- 部署提示 ---
st.markdown("---")
st.caption("部署提示：请确保 GitHub 仓库中有 requirements.txt (包含 streamlit, opencv-python-headless, numpy, Pillow, easyocr)")
