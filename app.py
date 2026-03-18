import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import zipfile
import io

# --- 页面配置 ---
st.set_page_config(page_title="表情包高级处理工具", layout="wide")
st.title("🎨 表情包精准处理 (连通域抠图版)")
st.info("说明：当前算法采用‘油漆桶’原理，从边缘向内识别背景，不会误伤人物脸部颜色。")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

# --- 侧边栏 ---
st.sidebar.header("🛠️ 核心参数")
m = st.sidebar.number_input("行数 (m)", min_value=1, value=4)
n = st.sidebar.number_input("列数 (n)", min_value=1, value=3)
remove_text = st.sidebar.checkbox("无痕去字", value=True)
# 容差值：决定油漆桶渗入颜色的程度
tolerance = st.sidebar.slider("背景识别容差", 1, 100, 20, help="值越大，越能处理有杂色的背景；值越小，边缘越精准。")

st.sidebar.markdown("---")
download_mode = st.sidebar.radio("📥 下载方式", ["ZIP 压缩包", "手动单张预览"])

def smart_remove_background(cv_img, tol):
    """
    使用 Flood Fill 算法从边缘抠图，保护主体内部。
    """
    h, w = cv_img.shape[:2]
    # 转换为填充专用的图
    flood_fill = cv_img.copy()
    
    # 创建掩码 (必须比原图大 2 像素)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 从四个角尝试填充 (假设角点是背景)
    seeds = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]
    for x, y in seeds:
        cv2.floodFill(flood_fill, mask, (x, y), (255, 255, 255), 
                      (tol, tol, tol), (tol, tol, tol), 
                      cv2.FLOODFILL_FIXED_RANGE)

    # mask 此时记录了所有被填充的区域 (背景 = 1, 其他 = 0)
    # 提取有效区域 (去掉多余的 2 像素边缘)
    final_mask = mask[1:-1, 1:-1] * 255 
    
    # 转换为 RGBA
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    
    # 反转蒙版：背景变透明 (255 -> 0)，人物保留 (0 -> 255)
    alpha = Image.fromarray(255 - final_mask).convert("L")
    pil_img.putalpha(alpha)
    
    return pil_img

# --- 主逻辑 ---
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="原图预览", use_container_width=True)

    if st.button("🚀 执行精准处理"):
        reader = load_ocr() if remove_text else None
        h, w, _ = img_bgr.shape
        tile_h, tile_w = h // m, w // n
        processed_images = []
        
        progress_bar = st.progress(0)
        total = m * n

        for r in range(m):
            for c in range(n):
                y_s, y_e = r*tile_h, (r+1)*tile_h
                x_s, x_e = c*tile_w, (c+1)*tile_w
                tile = img_bgr[y_s:y_e, x_s:x_e].copy()

                # 1. 去字 (在抠图前做，防止文字干扰背景识别)
                if remove_text:
                    results = reader.readtext(tile)
                    text_mask = np.zeros(tile.shape[:2], dtype=np.uint8)
                    for (bbox, text, prob) in results:
                        if any('\u4e00' <= char <= '\u9fff' for char in text):
                            pts = np.array(bbox, np.int32)
                            cv2.fillPoly(text_mask, [pts], 255)
                    if np.any(text_mask):
                        tile = cv2.inpaint(tile, text_mask, 3, cv2.INPAINT_NS)

                # 2. 精准抠图 (Flood Fill)
                pil_img = smart_remove_background(tile, tolerance)
                processed_images.append((pil_img, f"sticker_{r+1}_{c+1}.png"))
                
                progress_bar.progress(len(processed_images) / total)

        st.success("✨ 处理完成！脸部颜色已成功保护。")

        # --- 下载区域 ---
        if download_mode == "ZIP 压缩包":
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED, False) as zf:
                for img, name in processed_images:
                    img_io = io.BytesIO()
                    img.save(img_io, format='PNG')
                    zf.writestr(name, img_io.getvalue())
            st.download_button("📥 下载 ZIP", zip_buf.getvalue(), "stickers.zip", "application/zip", use_container_width=True)
        else:
            cols = st.columns(3)
            for i, (img, name) in enumerate(processed_images):
                with cols[i % 3]:
                    st.image(img, caption=name)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(f"下载 {name}", buf.getvalue(), name, "image/png", key=f"d_{i}")
