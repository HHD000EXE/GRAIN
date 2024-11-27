import csv
from PIL import Image, ImageDraw

coordinates_list = [(64, 72), (72, 72), (80, 72), (88, 72), (96, 72), (64, 80), (72, 80), (80, 80), (88, 80), (96, 80), (64, 88), (72, 88), (80, 88), (88, 88), (96, 88)]  # Action position

# Function to generate an image with a white rectangle at the specified index
def generate_image(index, coordinates_list):
    if index >= len(coordinates_list):
        print(f"Index {index} is out of bounds for the coordinates list.")
        return

    width, height = 128, 128  # Image dimensions
    rectangle_size = 20  # Size of the rectangle window

    # Create a black image
    img = Image.new('1', (width, height), color=0)  # '1' for 1-bit pixels (black and white)

    draw = ImageDraw.Draw(img)
    center_x, center_y = coordinates_list[index]

    # Calculate rectangle coordinates
    left = center_x - rectangle_size // 2
    top = center_y - rectangle_size // 2
    right = center_x + rectangle_size // 2
    bottom = center_y + rectangle_size // 2

    # Draw a white rectangle
    draw.rectangle([left, top, right, bottom], fill=1)

    return img

# Read the CSV file and process the column of integers
def process_csv(file_path, column_name, coordinates_list):
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            index = int(row[column_name])
            f_name = row['Image Name']
            img = generate_image(index, coordinates_list)
            if img:
                # Save the image or display it
                img.save(f'Test_data/train_images(Action)/{f_name[:-4]}_action.png')

# Example usage
file_path = 'Test_data/label/label.csv'  # Path to your CSV file
column_name = 'Action'  # Name of the column with integers

process_csv(file_path, column_name, coordinates_list)
