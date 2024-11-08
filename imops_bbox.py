import os
import json
import fitz
import cv2
import numpy as np

def draw_coordinates(json_path, pdf_base_path, save_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for idx in range(len(data)):
        pdf_name = f"{data[idx]['metadata']['file_name']}.pdf"
        page_num = data[idx]['metadata']['page']
        coordinates = data[idx]['metadata']['coordinates']

        pdf_path = os.path.join(pdf_base_path, pdf_name)

        pdf_document = fitz.open(pdf_path)
        page = pdf_document[data[idx]['metadata']['page'] - 1]

        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()

        height, width = img.shape[:2]

        for coord in coordinates:
            x1 = int(coord['x'] * width)
            y1 = int(coord['y'] * height)

            if coord == coordinates[-1]:
                next_coord = coordinates[0]
            else:
                next_coord = coordinates[coordinates.index(coord) + 1]
            
            x2 = int(next_coord['x'] * width)
            y2 = int(next_coord['y'] * height)

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

        output_path = f"{save_dir}/bbox_images/{idx}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, img)
        pdf_document.close()