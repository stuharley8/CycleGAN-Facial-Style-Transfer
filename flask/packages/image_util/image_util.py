import cv2

def export_image_to_file(export_image_name, image_matrix):
    cv2.imwrite(f'{export_image_name}.jpg', image_matrix)