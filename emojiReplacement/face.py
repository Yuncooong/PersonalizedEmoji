import cv2
import numpy as np
from face_detect import find_faces


def image_as_nparray(image):
    return np.asarray(image)

def draw_with_alpha(source_image, image_to_draw, coordinates):
    """
    Draws a partially transparent image over another image.
    :param source_image: Image to draw over.
    :param image_to_draw: Image to draw.
    :param coordinates: Coordinates to draw an image at. Tuple of x, y, width and height.
    """
    x, y, w, h = coordinates
    image_array = image_as_nparray(image_to_draw)
    #print(image_array.shape)
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)


def main(model,image_path):
    emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    # Read the image
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Draw a rectangle around the faces
    for normalized_face, (x, y, w, h) in find_faces(image):
        #print((x, y, w, h))
        emotion = model.predict(normalized_face)  # do prediction
        emotion = emotion[0]
        print emotions[emotion]
        # try to add a emoji to a existing face
        s_img = cv2.imread("./emojis/"+emotions[emotion]+".png",cv2.IMREAD_UNCHANGED)
        s_height, s_width, s_channels = s_img.shape
        l_img = image

        resize_ratio = round(w*10/(10.0*s_width), 5)

        s_img = cv2.resize(s_img, (0,0), fx= resize_ratio, fy= resize_ratio)

        draw_with_alpha(l_img,s_img,(x, y, w, h))
    cv2.imwrite("result-emoji.jpg", image)
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    image_path = "./face_img/4.jpg"
    fisher_face = cv2.createFisherFaceRecognizer()
    fisher_face.load('models/emotion_detection_model.xml')
    main(fisher_face,image_path)
