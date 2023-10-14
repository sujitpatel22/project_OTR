import cv2

def gen_contours(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
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
