import FaceDetector as fd
import os
import glob
import cv2
import shutil

faces = []
rejected = []

model = fd.FaceDetector('faceDetectionModel/deploy.prototxt.txt', 'faceDetectionModel/opencv_face_detector.caffemodel')


for image in glob.glob("images/*.jpg"):
    face = model.detect_face_from_image(image, 0.5)
    if (face is None):
        rejected.append(image)   
    else:
        faces.append(face)


# for image in glob.glob("images/*.jpg"):
#     face = model.detect_face_from_image(image, 0.8)
#     if (face is None):
#         rejected.append(image)   
#     else:
#         faces.append(face)


for i, face in enumerate(faces):
    try:
        if len(face) > 0:
            cv2.imwrite(f'faces/face{i}.jpg', face)
    except cv2.error:
        print(face)

for i, rejected in enumerate(rejected):
    shutil.copy2(rejected, f'rejected')

# model.detect_face_from_image('images/1_0_1_20170117130048013.jpg')
# cv2.imwrite(f'faces/test.jpg', model.detect_face_from_image('images/1_0_1_20170117130048013.jpg'))