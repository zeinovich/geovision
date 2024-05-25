from PIL import Image
import cv2
from pdf2image import convert_from_bytes

def load_and_display_file(uploaded_file):
    file_type = uploaded_file.type.split('/')[-1].lower()

    if file_type in ['jpg', 'jpeg', 'png', 'tiff', 'tif']:
        image = Image.open(uploaded_file)
    elif file_type == 'pdf':
        # Используем pdf2image для конвертации PDF в изображение
        images = convert_from_bytes(uploaded_file.read())
        image = images[0]  # Берем первую страницу PDF
    elif file_type == 'cdr':
        return None
    else:
        st.error("Неподдерживаемый формат файла")
        return None

    return image