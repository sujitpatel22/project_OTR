from tensorflow.keras.models import load_model
import numpy as np
import pygame
from PIL import Image, ImageDraw
import sys

# if len(sys.argv) != 2:
#     raise Exception("Error: missing model instance!")
#     # sys.exit(1)

# model = load_model(sys.argv[1])

def main():
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
                pygame.draw.line(canvas, (0,0,0), last_pos, event.pos, 20)
                last_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pygame.image.save(screen, "sample.png")
                    # img = Image.open("sample.png")

                    # word = recogniser(img)
                    # print(word)

                elif event.key == pygame.K_BACKSPACE:
                    canvas.fill((255, 255, 255))
                elif event.key == pygame.K_q:
                    running = False

        screen.blit(canvas, (0, 0))
        pygame.display.update()


def recogniser(img):
    img = np.array(img).reshape(1, 28, 28, 1)
    predicted_label = model.predict(img).argmax()
    predicted_character = chr(predicted_label)
    return predicted_character


if __name__ == "__main__":
    main()