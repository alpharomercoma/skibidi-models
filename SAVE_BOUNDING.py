import os
import csv
import json
from PIL import Image, ImageColor, ImageDraw
from pydantic import BaseModel

class BoundingBox(BaseModel):
    """
    Represents a bounding box with its 2D coordinates.

    Attributes:
        box_2d (list[float]): Coordinates in normalized [y1, x1, y2, x2] format (values between 0 and 1).
    """
    box_2d: list[float]

def plot_bounding_boxes(image_uri: str, bounding_boxes: list[BoundingBox]) -> None:
    """
    Draws bounding boxes on the image and saves the output to ./new_dataset/bb/.

    Args:
        image_uri (str): Path to the input image.
        bounding_boxes (list[BoundingBox]): List of bounding boxes.
    """
    with Image.open(image_uri) as im:
        width, height = im.size
        draw = ImageDraw.Draw(im)
        colors = list(ImageColor.colormap.keys())

        for i, bbox in enumerate(bounding_boxes):
            y1, x1, y2, x2 = bbox.box_2d
            # Convert normalized coordinates (0 to 1) to absolute pixel values.
            abs_y1 = int(y1 * height)
            abs_x1 = int(x1 * width)
            abs_y2 = int(y2 * height)
            abs_x2 = int(x2 * width)
            color = colors[i % len(colors)]
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Ensure the output directory exists.
        output_dir = './new_dataset/bb/'
        os.makedirs(output_dir, exist_ok=True)
        # Use the same filename as the input image.
        filename = os.path.basename(image_uri)
        output_path = os.path.join(output_dir, filename)
        im.save(output_path)
        print(f"Saved image with bounding boxes to {output_path}")

if __name__ == '__main__':
    csv_file_path = '/home/alpha/alpha/alpha/new_dataset/bounding_box.csv'

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_uri = row['visual_path']
            # The bounding_boxes column is a JSON string; parse it.
            bounding_boxes_str = row['bounding_boxes']
            bbox_list = json.loads(bounding_boxes_str)
            bounding_boxes = [BoundingBox(box_2d=box) for box in bbox_list]
            plot_bounding_boxes(image_uri, bounding_boxes)
