import xml.etree.ElementTree as ET
import os

# Paths
annotations_dir = "dataset/annotations/train"
images_dir = "dataset/images/train"
labels_dir = "dataset/labels/train"

os.makedirs(labels_dir, exist_ok=True)

class_map = {
    "person": 0
}

for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(annotations_dir, xml_file))
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(images_dir, image_name)

    if not os.path.exists(image_path):
        continue

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    label_file = os.path.join(labels_dir, image_name.replace(".jpg", ".txt"))

    with open(label_file, "w") as f:
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in class_map:
                continue

            xml_box = obj.find("bndbox")
            xmin = float(xml_box.find("xmin").text)
            ymin = float(xml_box.find("ymin").text)
            xmax = float(xml_box.find("xmax").text)
            ymax = float(xml_box.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            f.write(f"0 {x_center} {y_center} {width} {height}\n")

print("✅ VOC → YOLO conversion completed")
