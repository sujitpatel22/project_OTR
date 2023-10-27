from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pygame
from PIL import Image, ImageDraw
import sys

IMG_WIDTH = 168
IMG_HEIGHT = 32
model = None

def load_model():
    model = load_model(sys.argv[1])

def main():
    if len(sys.argv) != 2:
        raise Exception("Error: missing model instance!")

    load_model()
    # load_canvas()

    img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
    word = recogniser(img)
    print(word)


def load_canvas():
    pygame.init()

    width, height = 1024, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("OCR canvas")

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
                pygame.draw.line(canvas, (0,0,0), last_pos, event.pos, 40)
                last_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # pygame.image.save(screen, "sample.png")
                    img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
                    word = recogniser(img)
                    print(word)

                elif event.key == pygame.K_BACKSPACE:
                    canvas.fill((255, 255, 255))
                elif event.key == pygame.K_q:
                    running = False

        screen.blit(canvas, (0, 0))
        pygame.display.update()


def recogniser(img):
    img = get_cropped(img)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    img = np.array(img).reshape(1, 28, 28, 1)

    predicted_label = model.predict(img).argmax()
    predicted_character = chr(predicted_label)
    return predicted_character


def get_cropped(img):    
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    x_min = min(box[0] for box in text_boxes)
    y_min = min(box[1] for box in text_boxes)
    x_max = max(box[0] + box[2] for box in text_boxes)
    y_max = max(box[1] + box[3] for box in text_boxes)
    
    cropped_image = img[y_min:y_max, x_min:x_max]
    cv2.imwrite(f"cropped_{image}.png", cropped_image)
    
    img = cv2.imread(f"cropped_{image}.png", cv2.IMREAD_GRAYSCALE)
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
