import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import zipfile
import io
import os

# 页面配置
st.set_page_config(page_title="表情包切割助手", layout="centered")
st.title("🎨 动漫表情包自动处理工具")
st.write("上传一张表情包大图，自动去字、切割、抠图并打包。")

# 侧边栏参数设置
st.sidebar.header("参数设置")
m = st.sidebar.number_input("行数 (m)", min_value=1, value=4)
n = st.sidebar.number_input("列数 (n)", min_value=1, value=3)
remove_text = st.sidebar.checkbox("去除中文", value=True)
threshold = st.sidebar.slider("抠图灵敏度", 0, 100, 30)

# 初始化 OCR (缓存模型防止重复加载)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

uploaded_file = st.file_uploader("选择表情包图片...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, caption="原始图片", use_container_width=True)

    if st.button("开始处理"):
        reader = load_ocr()
        h, w, _ = img.shape
        tile_h, tile_w = h // m, w // n
        
        # 准备 ZIP 内存缓冲区
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            progress_bar = st.progress(0)
            
            for r in range(m):
                for c in range(n):
                    # 1. 切割
                    tile = img[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
                    
                    # 2. 去字 (简化逻辑)
                    if remove_text:
                        results = reader.readtext(tile)
                        mask = np.zeros(tile.shape[:2], dtype=np.uint8)
                        for (bbox, text, prob) in results:
                            if any('\u4e00' <= char <= '\u9fff' for char in text):
                                pts = np.array(bbox, np.int32)
                                cv2.fillPoly(mask, [pts], 255)
                        tile = cv2.inpaint(tile, mask, 3, cv2.INPAINT_NS)

                    # 3. 抠图转透明
                    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(tile_rgb).convert("RGBA")
                    data = pil_img.getdata()
                    bg_color = data[0] # 取左上角为背景色
                    
                    new_data = []
                    for item in data:
                        diff = sum(abs(item[i] - bg_color[i]) for i in range(3))
                        new_data.append((0,0,0,0) if diff < threshold else item)
                    pil_img.putdata(new_data)
                    
                    # 4. 存入 ZIP
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG')
                    zip_file.writestr(f"sticker_{r}_{c}.png", img_byte_arr.getvalue())
                
                progress_bar.progress((r + 1) / m)

        st.success("处理完成！")
        st.download_button(
            label="点击下载 ZIP 压缩包",
            data=zip_buffer.getvalue(),
            file_name="stickers.zip",
            mime="application/zip"
        )
