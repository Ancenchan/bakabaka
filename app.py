import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import zipfile
import io

# --- 页面配置 ---
st.set_page_config(page_title="表情包高级处理工具", layout="wide")
st.title("🎨 表情包精准处理 (状态记忆版)")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

# --- 初始化 Session State (状态保持) ---
if 'processed_images' not in st.session_state:
    st.session_state['processed_images'] = []
if 'process_done' not in st.session_state:
    st.session_state['process_done'] = False

# --- 侧边栏 ---
st.sidebar.header("🛠️ 核心参数")
m = st.sidebar.number_input("行数 (m)", min_value=1, value=4)
n = st.sidebar.number_input("列数 (n)", min_value=1, value=3)
remove_text = st.sidebar.checkbox("无痕去字", value=True)
tolerance = st.sidebar.slider("背景识别容差", 1, 100, 20)

st.sidebar.markdown("---")
download_mode = st.sidebar.radio("📥 下载方式", ["ZIP 压缩包", "手动单张预览"])

def smart_remove_background(cv_img, tol):
    h, w = cv_img.shape[:2]
    flood_fill = cv_img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 从四个角尝试填充
    seeds = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]
    for x, y in seeds:
        cv2.floodFill(flood_fill, mask, (x, y), (255, 255, 255), 
                      (tol, tol, tol), (tol, tol, tol), 
                      cv2.FLOODFILL_FIXED_RANGE)

    final_mask = mask[1:-1, 1:-1] * 255 
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    alpha = Image.fromarray(255 - final_mask).convert("L")
    pil_img.putalpha(alpha)
    return pil_img

# --- 主逻辑 ---
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])

# 如果上传了新文件，重置之前的处理状态
if uploaded_file:
    if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded_file.name:
        st.session_state['processed_images'] = []
        st.session_state['process_done'] = False
        st.session_state['last_uploaded'] = uploaded_file.name

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # 只有在没处理过的情况下才显示原图预览
    if not st.session_state['process_done']:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="原图预览", use_container_width=True)

    if st.button("🚀 执行精准处理"):
        reader = load_ocr() if remove_text else None
        h, w, _ = img_bgr.shape
        tile_h, tile_w = h // m, w // n
        
        temp_results = []
        progress_bar = st.progress(0)
        total = m * n

        for r in range(m):
            for c in range(n):
                y_s, y_e = r*tile_h, (r+1)*tile_h
                x_s, x_e = c*tile_w, (c+1)*tile_w
                tile = img_bgr[y_s:y_e, x_s:x_e].copy()

                if remove_text:
                    results = reader.readtext(tile)
                    text_mask = np.zeros(tile.shape[:2], dtype=np.uint8)
                    for (bbox, text, prob) in results:
                        if any('\u4e00' <= char <= '\u9fff' for char in text):
                            pts = np.array(bbox, np.int32)
                            cv2.fillPoly(text_mask, [pts], 255)
                    if np.any(text_mask):
                        tile = cv2.inpaint(tile, text_mask, 3, cv2.INPAINT_NS)

                pil_img = smart_remove_background(tile, tolerance)
                
                # 将 PIL 转为字节流，方便 session_state 存储和下载
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                temp_results.append({"img": pil_img, "bytes": buf.getvalue(), "name": f"sticker_{r+1}_{c+1}.png"})
                
                progress_bar.progress(len(temp_results) / total)
        
        # 存入 session_state
        st.session_state['processed_images'] = temp_results
        st.session_state['process_done'] = True
        st.rerun() # 处理完强制刷新一次以显示结果

    # --- 显示和下载区域 (从 session_state 读取) ---
    if st.session_state['process_done']:
        st.success("✨ 处理完成！你可以连续下载，图片不会再消失了。")
        
        if download_mode == "ZIP 压缩包":
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED, False) as zf:
                for item in st.session_state['processed_images']:
                    zf.writestr(item['name'], item['bytes'])
            
            st.download_button("📥 下载全集 ZIP", zip_buf.getvalue(), "stickers.zip", "application/zip", use_container_width=True)
        
        # 无论选哪种模式，都展示预览，方便用户查看
        st.write("### 📸 处理结果预览")
        cols = st.columns(3)
        for i, item in enumerate(st.session_state['processed_images']):
            with cols[i % 3]:
                st.image(item['img'], caption=item['name'])
                st.download_button(
                    label=f"保存 {item['name']}",
                    data=item['bytes'],
                    file_name=item['name'],
                    mime="image/png",
                    key=f"dl_{i}" # 唯一的 key 保证按钮独立
                )
