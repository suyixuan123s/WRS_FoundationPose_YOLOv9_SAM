import cv2
import numpy as np

def count_and_show_sizes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查路径是否正确。")
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cigarette_box_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5513:
            cigarette_box_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(f"烟盒 {cigarette_box_count}: 面积 = {area}, 尺寸 = {w} x {h}")
    print(f"检测到的烟盒数量: {cigarette_box_count}")
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Cigarette Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cigarette_box_count


image_path = 'demo.jpg'
count_and_show_sizes(image_path)
