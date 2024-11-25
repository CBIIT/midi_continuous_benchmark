####################################
# The script is to compare the pixel image between before and after DICOM de-identification, and the save the comparison result in a pdf file
# Developer: Qinyan Pan
# Modified Date: 2024-11-25 by Linmin Pei
####################################
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import measure, morphology
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import csv
import os
import textwrap
import PyPDF2
import shutil
import json
import logging
import time
import argparse
import math

# Define output_pdf as a global variable
OUTPUT_PDF = 'comparison_results.pdf'
# Using global debug flag
DEBUG_MODE = False


def load_dicom_image(file_path):
    """Load a DICOM file and return the image data as a numpy array."""
    try:
        dicom = pydicom.dcmread(file_path)
        # Check if Pixel Data is present
        if 'PixelData' not in dicom:
            return None
        image = dicom.pixel_array
        return image
    except AttributeError as e:
        print(f"Error loading DICOM image from {file_path}: {e}")
        return None
    except ValueError as e:
        print(f"ValueError loading DICOM image from {file_path}: {e}")
        return None


def compare_images(image1, image2):
    """Compare two images using Mean Squared Error and SSIM."""
    # Check image dimensions
    height, width = image1.shape[:2]

    # Set win_size to be smaller than the smallest dimension, and it must be odd
    win_size = min(height, width)
    if win_size % 2 == 0:
        win_size -= 1  # Ensure it's an odd number

    # Compute the data range from the image
    data_range = np.max(image1) - np.min(image1)

    mse = np.mean((image1 - image2) ** 2)
    ssim_index, diff = ssim(image1, image2, data_range=data_range,
                            win_size=win_size, channel_axis=-1, full=True)
    return mse, ssim_index, diff


def find_changed_area(image1, image2, threshold=30):
    """Find the area of the image that has changed."""
    diff = np.abs(image1 - image2)
    mask = diff > threshold
    changed_area = np.sum(mask)
    return changed_area, mask


def wrap_text(text, width):
    """Wrap text to a specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def process_batch(batch_dir1, batch_dir2, output_pdf):
    """Compare images in two batches of DICOM files and save results to a PDF."""
    batch1_files = sorted(
        [f for f in os.listdir(batch_dir1) if f.endswith('.dcm')])
    batch2_files = sorted(
        [f for f in os.listdir(batch_dir2) if f.endswith('.dcm')])

    with PdfPages(output_pdf) as pdf:
        for file1, file2 in zip(batch1_files, batch2_files):
            file_path1 = os.path.join(batch_dir1, file1)
            file_path2 = os.path.join(batch_dir2, file2)

            image1 = load_dicom_image(file_path1)
            image2 = load_dicom_image(file_path2)

            mse, ssim_index, diff = compare_images(image1, image2)
            changed_area, mask = find_changed_area(image1, image2)

            if mse > 0 or ssim_index < 1:
                print("Significant changes detected.")
                print(f"Comparing {file_path1} and {file_path2}:")
                print(f"MSE: {mse}, SSIM: {ssim_index}")
            else:
                print("Images are similar.")
                return

            # Create a figure with the original images and the difference image with marked changes
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image1, cmap='gray')
            ax[0].set_title('Image 1')
            ax[1].imshow(image2, cmap='gray')
            ax[1].set_title('Image 2')
            ax[2].imshow(mask, cmap='gray')
            ax[2].set_title('Changed Areas')

            # Wrap the text to fit the page width
            title_text = f"Comparison: {file_path1} vs {file_path2}"
            wrapped_title = wrap_text(title_text, width=120)
            metrics_text = f"MSE: {mse:.2f}, SSIM: {ssim_index:.2f}"

            # Add wrapped title and metrics text to the figure
            # Adjust the top space to fit the title
            plt.subplots_adjust(top=0.7)
            plt.suptitle(f"{wrapped_title}\n{metrics_text}\n", fontsize=12)

            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)


def convert_image_data(image):
    """
    Convert image data to a supported type for matplotlib.

    Parameters:
    - image: np.ndarray
        The input image array.

    Returns:
    - np.ndarray
        The converted image array in float32 format.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Check the current data type of the image
    current_dtype = image.dtype
    print(f"Current image dtype: {current_dtype}")

    # Convert image to float32 if it's not already in float32 or float64
    if current_dtype in [np.float32, np.float64]:
        return image
    elif np.issubdtype(current_dtype, np.integer):
        return image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype: {current_dtype}")


def save_figure_to_temp_pdf(fig, temp_pdf):
    """Save the figure to a temporary PDF file."""
    for ax in fig.axes:
        for im in ax.get_images():
            im.set_data(convert_image_data(im.get_array()))

    with PdfPages(temp_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def append_to_first_pdf(pdf1_path, pdf2_path):
    # Open the first PDF file in read mode
    with open(pdf1_path, 'rb') as pdf1_file:
        # Create a PDF reader object for the first PDF
        pdf1_reader = PyPDF2.PdfReader(pdf1_file)

        # Open the second PDF file in read mode
        with open(pdf2_path, 'rb') as pdf2_file:
            # Create a PDF reader object for the second PDF
            pdf2_reader = PyPDF2.PdfReader(pdf2_file)

            # Create a PDF writer object
            pdf_writer = PyPDF2.PdfWriter()

            # Add all pages from the first PDF to the writer
            for page_num in range(len(pdf1_reader.pages)):
                page = pdf1_reader.pages[page_num]
                pdf_writer.add_page(page)

            # Add all pages from the second PDF to the writer
            for page_num in range(len(pdf2_reader.pages)):
                page = pdf2_reader.pages[page_num]
                pdf_writer.add_page(page)

            # Write the combined pages to the first PDF file
            with open(pdf1_path, 'wb') as output_pdf:
                pdf_writer.write(output_pdf)


def display_image(ax, image, cmap='gray', error_message=None):
    """
    Display an image on a matplotlib axis and handle errors.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axis to display the image on.
    - image: np.ndarray
        The image array to be displayed.
    - cmap: str
        The colormap to use for grayscale images.
    - error_message: str
        Optional message to add to the PDF if image display fails.
    """
    try:
        ax.imshow(image, cmap=cmap)
    except Exception as e:
        print(f"Error displaying image: {e}")
        if error_message:
            # Add a message to the PDF if image display fails
            ax.text(0.5, 0.5, error_message, transform=ax.transAxes,
                    fontsize=12, color='red', ha='center', va='center')


def compare_dicom_image(file_path1, file_path2, output_pdf):
    image1 = load_dicom_image(file_path1)
    image2 = load_dicom_image(file_path2)

    if image1 is None or image2 is None:
        return

    if image1.size == 0 or image2.size == 0:
        return

    # Check if images have the expected number of dimensions
    if len(image1.shape) < 2 or len(image2.shape) < 2:
        return

    # Ensure images have the same dimensions
    if image1.shape != image2.shape:
        return

    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)

    mse, ssim_index, diff = compare_images(image1, image2)

    try:
        ssim_index = float(ssim_index)
    except ValueError:
        print(
            f"SSIM is invalid (not a number) when comparing {file_path1} and {file_path2}, so skipping the comparison.")
        print(f"SSIM: {ssim_index}, Type: {type(ssim_index)}")
        return

    if math.isnan(ssim_index):
        print(
            f"\n\nSSIM is NaN when compare {file_path1} and {file_path2} so skip the comparision")
        return

        # If images are similar, skip processing and exit the function
    if mse == 0 and ssim_index == 1:
        print(
            f"\n\nThis is not structure change in the pixel image between before and after de-identification, or possible wrong mapping file")
        return

    error_message1 = "The format of this image cannot be displayed."
    error_message2 = "The format of this image cannot be displayed."

    changed_area, mask = find_changed_area(image1, image2)
    mask = np.array(mask, dtype=np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    display_image(ax[0], image1, cmap='gray', error_message=error_message1)

    ax[0].set_title('De-identified image')
    display_image(ax[1], image2, cmap='gray', error_message=error_message2)
    ax[1].set_title('Source image')
    display_image(ax[2], mask, cmap='gray', error_message=error_message2)
    ax[2].set_title('Changed Areas')

    # Wrap the text to fit the page width
    title_text = f"Comparison: {file_path1} vs {file_path2}"
    wrapped_title = wrap_text(title_text, width=120)
    metrics_text = f"MSE: {mse:.2f}, SSIM: {ssim_index:.2f}"

    # Add wrapped title and metrics text to the figure
    plt.subplots_adjust(top=0.7)  # Adjust the top space to fit the title
    plt.suptitle(f"{wrapped_title}\n{metrics_text}\n", fontsize=12)

    # Save the figure to the temporary PDF
    temp_pdf = 'temp_output.pdf'
    save_figure_to_temp_pdf(fig, temp_pdf)

    # Check if the output PDF does not exist or is empty, copy the temporary PDF to the output PDF
    if not os.path.exists(output_pdf) or os.path.getsize(output_pdf) == 0:
        shutil.copy(temp_pdf, output_pdf)
        os.remove(temp_pdf)
        return
    else:
        append_to_first_pdf(output_pdf, temp_pdf)
        os.remove(temp_pdf)


def build_dicom_map(folder_path):
    dicom_map = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                ds = pydicom.dcmread(file_path)
                sop_instance_uid = ds.SOPInstanceUID
                dicom_map[sop_instance_uid] = file_path
    return dicom_map


def read_id_map(csv_file):
    id_map = {}
    print(f"mapping file: {csv_file}")

    with open(csv_file, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        print("Headers:", reader.fieldnames)  # Debugging line

        # Clean headers
        reader.fieldnames = [name.lstrip('\ufeff').strip()
                             for name in reader.fieldnames]

        for row in reader:
            id_map[row['id_new']] = row['id_old']
    return id_map


def compare_two_batch_DICOMs(new_map, old_map, id_map, logger, output_pdf):
    counter = 0
    for sop_instance_uid in new_map:
        id_new = sop_instance_uid
        new_file_path = new_map.get(id_new)

        id_old = id_map.get(id_new)
        if id_old:
            old_file_path = old_map.get(id_old)
            if old_file_path:
                counter += 1
                compare_dicom_image(new_file_path, old_file_path, output_pdf)
            else:
                print(
                    f"!!!!!!!not find match old_file_path for id_old ={id_old}")
        else:
            print(f"########not finding matching id_old for id_new={id_new}")


def debug_print(message):
    if DEBUG_MODE:
        print(message)


def main(config_file):
    # Read configuration from config.json
    with open(config_file) as config_file:
        config = json.load(config_file)

# Paths to folders and files
    new_folder_path = config['input_data_path']
    old_folder_path = config['pre_deID_data_path']
    id_map_csv_file = config['uid_mapping_file']
    output_pdf = config['output_data_path']
    run_name = config['run_name']
    output_pdf = os.path.join(output_pdf, run_name, OUTPUT_PDF)

# Ensure the directory exists
    log_directory = config['log_path']
    os.makedirs(log_directory, exist_ok=True)

# Configure logging to write to a specific location
    logging.basicConfig(
        filename=os.path.join(log_directory, 'imgCmpLog.log'),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger()

# Start the timer
    start_time = time.time()

    new_map = build_dicom_map(new_folder_path)
    old_map = build_dicom_map(old_folder_path)
    id_map = read_id_map(id_map_csv_file)
    compare_two_batch_DICOMs(new_map, old_map, id_map, logger, output_pdf)
    if os.path.exists(output_pdf) == False:
        # create a pdf file showing information of no much change in pixel image before and after de-identification
        c = canvas.Canvas(output_pdf, pagesize=letter)
        c.drawString(
            100, 750, "This is not structure change in the pixel image between before and after de-identification, or caused by possible incorrect mapping file.")
        c.save()

# Stop the timer
    end_time = time.time()

# Calculate and display the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    debug_print("Execution time: " + str(execution_time) + " seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparing DICOM files based on a configuration file")
    parser.add_argument('config', help="Path to the configuration JSON file")

    args = parser.parse_args()
    main(args.config)
