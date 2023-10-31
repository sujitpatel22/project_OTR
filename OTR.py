from tensorflow.keras.models import load_model
from CRNN_engine3 import ModelGenerator
import cv2
import numpy as np
import pygame
from PIL import Image, ImageDraw
import sys

IMG_HEIGHT = 32
IMG_WIDTH = 170
MG = ModelGenerator(170, 32, n_classes=37, 20)

def fetch_model():
    model = MG.build_model("save")
    model = MG.compile(model = model)
    model.load_weights(sys.argv[1])
    return model

def main():
    if len(sys.argv) != 2:
        raise Exception("Error: missing model instance!")

    model = fetch_model()
    load_canvas(model)


def load_canvas(model):
    pygame.init()

    width, height = 1024, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("OTR canvas")

    canvas = pygame.Surface((width, height))
    canvas.fill((255, 255, 255))

    running = True
    drawing = False
    last_pos = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.line(canvas, (0,0,0), last_pos, event.pos, 25)
                last_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pygame.image.save(screen, "temp/sample.png")
                    img = cv2.imread("temp/sample.png", cv2.IMREAD_GRAYSCALE)
                    word = recogniser(model, img)
                    print(word)

                elif event.key == pygame.K_BACKSPACE:
                    canvas.fill((255, 255, 255))
                elif event.key == pygame.K_q:
                    running = False

        screen.blit(canvas, (0, 0))
        pygame.display.update()


def recogniser(model, img):
    img = get_cropped(img)
    img = resize(img)
    cv2.imwrite("temp/resized_img.png", img)
    img = np.expand_dims(img, axis = -1)
    img = np.expand_dims(img, axis = 0)
    img = img / 255.0

    predicted_label = model.predict(img)
    predicted_label = MG.convert_label(pred_label = predicted_label)
    predicted_label = MG.decode_label(label = predicted_label)
    return predicted_label

def resize(image):
    scale_x = IMG_WIDTH / image.shape[1]
    scale_y = IMG_HEIGHT / image.shape[0]
    scale = min(scale_x, scale_y)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pad_x = max(0, (IMG_WIDTH - image.shape[1]) // 2)
    pad_y = max(0, (IMG_HEIGHT - image.shape[0]) // 2)
    image = cv2.copyMakeBorder(image, pad_y, IMG_HEIGHT - image.shape[0] - pad_y,
                               pad_x, IMG_WIDTH - image.shape[1] - pad_x, cv2.BORDER_CONSTANT, value=255)
    return image


def get_cropped(img):
    _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_boxes = [cv2.boundingRect(contour) for contour in contours]
    x_min = min(box[0] for box in text_boxes)
    y_min = min(box[1] for box in text_boxes)
    x_max = max(box[0] + box[2] for box in text_boxes)
    y_max = max(box[1] + box[3] for box in text_boxes)
    cropped_image = img[y_min:y_max, x_min:x_max]
    # cv2.imwrite("temp/cropped_temp.png", cropped_image)
    _, img = cv2.threshold(cropped_image, 180, 255, cv2.THRESH_BINARY_INV)
    return img
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented_words = []
    img_num = 1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        word_image = img[y:y+h, x:x+w]
        word_image = cv2.bitwise_not(word_image)
        segmented_words.append(word_image)
        cv2.imwrite(f"{image}_contour_{img_num}.png", word_image)
        img_num += 1

if __name__ == "__main__":
    main()
