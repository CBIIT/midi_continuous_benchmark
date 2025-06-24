###################
# The script is to compare the pixel images before and after DICOM de-identification, and save the comparison result in a pdf file
# Develop: Qinyan Pan
# Modification: Linmin Pei, on Dec. 27, 2024
##################
import pydicom
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import measure, morphology
from matplotlib.backends.backend_pdf import PdfPages
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
import time
from typing import Dict, Tuple
from pathlib import Path
from datetime import datetime


class MapProcessor(threading.Thread):
    def __init__(self, thread_id: int, sub_map: Dict[str, str], old_map, id_map, output_dir: str):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.sub_map = sub_map
        self.old_map = old_map
        self.id_map = id_map
        self.output_filename = Path(output_dir) / \
            f"comparsion_result_{thread_id}.pdf"
        self.start_time = None
        self.end_time = None
        self.execution_time = None

    def run(self):
        self.start_time = time.time()
        print(f"Starting Thread-{self.thread_id}")
        self.compare_two_batch_DICOMs()
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print(
            f"Finished Thread-{self.thread_id} in {self.execution_time:.2f} seconds")

    def compare_dicom_image(self, file_path1, file_path2, output_pdf):
        image1 = load_dicom_image(file_path1)
        image2 = load_dicom_image(file_path2)

        if image1 is None or image2 is None:
            print(
                f"\n@@@Skipping comparison due to invalid DICOM images when comparing {file_path1} and {file_path2}.")
            return

        if image1.shape[0] < 7 or image1.shape[1] < 7 or image2.shape[0] < 7 or image2.shape[1] < 7:
            print(
                f"\n!!!Images are too small for SSIM comparison when comparing {file_path1} and {file_path2}.")
            return

        if image1.size == 0 or image2.size == 0:
            print(
                f"\n$$$image size is 0 in one of the file: {file_path1} and {file_path2}.")
            return

        # Check if images have the expected number of dimensions
        if len(image1.shape) < 2 or len(image2.shape) < 2:
            print(
                f"\n###One of the images does not have enough dimensions: {file_path1} and {file_path2}.")
            return

        # Ensure images have the same dimensions
        if image1.shape != image2.shape:
            print(
                f"\n***Images have different dimensions: {file_path1} and {file_path2}.")
            return

        mse, ssim_index, diff = self.compare_images(image1, image2)

        try:
            ssim_index = float(ssim_index)
        except ValueError:
            print(
                f"\n!@#SSIM is invalid (not a number) when comparing {file_path1} and {file_path2}, so skipping the comparison.")
            print(f"SSIM: {ssim_index}, Type: {type(ssim_index)}")
            return

        if math.isnan(ssim_index):
            print(
                f"\n\nSSIM is NaN when compare {file_path1} and {file_path2} so skip the comparision")
            return

        if mse == 0 and ssim_index == 1:
            return

        error_message1 = "The format of this image cannot be displayed."
        error_message2 = "The format of this image cannot be displayed."

        thread_number = self.thread_id
        changed_area, mask = self.find_changed_area(image1, image2)

        mask = np.array(mask, dtype=image1.dtype)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        print(
            f"\n thread {thread_number} -- $$$Comparing {file_path1} and {file_path2}:")
        print(f"MSE: {mse}, SSIM: {ssim_index}")

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        self.display_image(ax[0], image1, cmap='gray',
                           error_message=error_message1)

        ax[0].set_title('Image After MIDI')
        self.display_image(ax[1], image2, cmap='gray',
                           error_message=error_message2)

        ax[1].set_title('Image Before MIDI')
        self.display_image(ax[2], mask_rgb, cmap='gray',
                           error_message=error_message2)

        ax[2].set_title('Changed Areas')

        # Wrap the text to fit the page width
        title_text = f"Comparison: {file_path1} vs {file_path2}"
        wrapped_title = self.wrap_text(title_text, width=120)
        metrics_text = f"MSE: {mse:.2f}, SSIM: {ssim_index:.2f}"

        # Add wrapped title and metrics text to the figure
        plt.subplots_adjust(top=0.7)  # Adjust the top space to fit the title
        plt.suptitle(f"{wrapped_title}\n{metrics_text}\n", fontsize=12)

        # Save the figure to the temporary PDF
        temp_pdf = f"temp_output-{thread_number}.pdf"
        self.save_figure_to_temp_pdf(fig, temp_pdf)

        # Check if the output PDF does not exist or is empty, copy the temporary PDF to the output PDF
        if not os.path.exists(output_pdf) or os.path.getsize(output_pdf) == 0:
            shutil.copy(temp_pdf, output_pdf)
            os.remove(temp_pdf)
            return
        else:
            self.append_to_first_pdf(output_pdf, temp_pdf)
            os.remove(temp_pdf)

    def compare_two_batch_DICOMs(self):
        new_map = self.sub_map
        old_map = self.old_map
        id_map = self.id_map
        output_pdf = self.output_filename
        thread_number = self.thread_id
        print(f"Starting compare two barch in thread {self.thread_id}")
        processed_uids = {}

        counter_for_progress = 0
        missing_matching_curated_file = 0
        for sop_instance_uid in new_map:
            id_new = sop_instance_uid
            counter_for_progress += 1

            if id_new not in processed_uids:
                new_file_path = new_map.get(id_new)

                id_old = id_map.get(id_new)
                if id_old:
                    old_file_path = old_map.get(id_old)
                    if old_file_path:
                        self.compare_dicom_image(
                            new_file_path, old_file_path, str(output_pdf))
                    else:
                        missing_matching_curated_file += 1
                        print(
                            f"!!!!!!!!error2--Not finding file path for ID_old :{id_old} -- in id map but does not exist.")
                else:
                    print(
                        f"!!!!!!!!error1--Not finding matching ID_old for id_new:{id_new}, that is in {new_file_path}")
            else:
                print(f"Skipping processed UID: {id_new}")

            if counter_for_progress % 100 == 0:
                print(
                    f"Thread-{thread_number}:{counter_for_progress}/{len(new_map)} processed")
        print(
            f"Thread-{thread_number}:Missing matching curated file count: {missing_matching_curated_file}")

    def compare_images(self, image1, image2):
        win_size = min(image1.shape[0], image1.shape[1],
                       image2.shape[0], image2.shape[1], 7) - 1
        if win_size % 2 == 0:  # Ensure win_size is odd
            win_size -= 1

        data_range = np.max(image1) - np.min(image1)

        mse = np.mean((image1 - image2) ** 2)
        ssim_index, diff = ssim(image1, image2, data_range=data_range,
                                win_size=win_size, channel_axis=-1, full=True)
        return mse, ssim_index, diff

    def find_changed_area(self, image1, image2, threshold=0):
        """Find the area of the image that has changed."""

        diff = np.abs(image1.astype(np.int16) - image2.astype(np.int16))

        if len(diff.shape) == 3:  # Color image
            mask = np.any(diff > threshold, axis=2)
        elif len(diff.shape) == 2:  # Grayscale image
            mask = diff > threshold
        else:
            raise ValueError("Unexpected image format")

        # Create a black background
        output_image = np.zeros_like(image1)

        if len(diff.shape) == 3:
            output_image[mask] = [0, 0, 255]  # Red for changes
        else:  # Grayscale case
            output_image = np.zeros_like(image1, dtype=np.uint8)
            output_image[mask] = 255  # Highlight changes in white

            # Convert grayscale output to BGR for visualization
            output_image = cv2.cvtColor(
                output_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return np.sum(mask), output_image

        # Count changed pixels
 #       changed_area = np.sum(mask)

        # Create a black background with the same size as the input images
 #       output_image = np.zeros_like(image1)

        # For color images, highlight changed areas in red
 #       if len(diff.shape) == 3:
 #           output_image[mask] = [0, 0, 255]  # Red for changes
 #       else:  # For grayscale images, convert to 3-channel for visualization
 #           output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
 #           output_image[mask] = [0, 0, 255]  # Red for changes

 #       output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

 #       return changed_area, output_image_rgb

    def wrap_text(self, text, width):
        """Wrap text to a specified width."""
        return '\n'.join(textwrap.wrap(text, width=width))

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
            # Convert integer types to float32
            return image.astype(np.float32)
        else:
            raise ValueError(f"Unsupported image dtype: {current_dtype}")

    def save_figure_to_temp_pdf(self, fig, temp_pdf):
        """Save the figure to a temporary PDF file."""
        # try:
    # Ensure the figure's images are in the correct format
    #    for ax in fig.axes:
    #        for im in ax.get_images():
    #            im.set_data(convert_image_data(im.get_array()))

        with PdfPages(temp_pdf) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        # except ExceptionType as e:
        #    print(f"@@@@error in create pdf file")

    def append_to_first_pdf(self, pdf1_path, pdf2_path):
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

    def display_image(self, ax, image, cmap='gray', error_message=None):
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


def build_dicom_map(folder_path, log_file):

    # Use the last directory name for the CSV file
    last_dir_name = os.path.basename(os.path.normpath(folder_path))
    csv_filename = f"ID_map_for_{last_dir_name}.csv"
    # csv_filename = uid_filepath_mapping

    if os.path.exists(csv_filename):
        print(f"mapping file exist: {csv_filename}")
        # If it exists, read the CSV file and return the map
        dicom_map = {}
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dicom_map[row['SOP instance uid']] = row['File path']
        return dicom_map

    print(f"create new mapping file: {csv_filename}")

    dicom_map = {}

    # Generate a default log file name if none is provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            os.getcwd(), f'dicom_processing_{timestamp}.log')

    # Ensure log_file is an absolute path
    log_file = os.path.abspath(log_file)

    # Set up logger for this processing run
    logger = setup_logger(log_file)

    logger.info(f"Starting processing for directory: {folder_path}")
    print(f"Starting processing for directory: {folder_path}")
    print(f"Log file location: {log_file}")

    duplicate_count = 0
    file_count = 0
    logger.info(f"Processing {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dcm'):
                file_count += 1
                file_path = os.path.join(root, file)
                ds = pydicom.dcmread(file_path)
                sop_instance_uid = ds.SOPInstanceUID

                if sop_instance_uid in dicom_map:
                    # Duplicate found, log and remove from map
                    duplicate_count += 1
                    duplicate_path = dicom_map[sop_instance_uid]
                    print(
                        f"Duplicate SOP Instance UID found: {sop_instance_uid}")
                    logger.info(
                        f"Duplicate SOP Instance UID found: {sop_instance_uid}")
                    logger.info(f"Duplicate file path: {file_path}")
                    logger.info(f"Conflict file path: {duplicate_path}")
                    del dicom_map[sop_instance_uid]
                else:
                    # Add to map
                    dicom_map[sop_instance_uid] = file_path

    # Write the map to a CSV file
    last_dir_name = os.path.basename(os.path.normpath(folder_path))
    csv_filename = f"ID_map_for_{last_dir_name}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
                                'SOP instance uid', 'File path'])
        writer.writeheader()
        for sop_instance_uid, file_path in dicom_map.items():
            writer.writerow(
                {'SOP instance uid': sop_instance_uid, 'File path': file_path})
    logger.info(f"Total files processed: {file_count}")
    logger.info(f"Duplicate SOPInstanceUID found: {duplicate_count}")
    logger.info(f"Total unique DICOM files: {len(dicom_map)}")
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

    # with open(csv_file, 'r') as file:
    #    reader = csv.DictReader(file)
        for row in reader:
            id_map[row['id_new']] = row['id_old']
    return id_map


def ensure_file_exists(filename):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['UID'])  # Write header
        print(f"Created new file: {filename}")


def load_processed_uids(filename):
    if not os.path.exists(filename):
        return set()
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        if 'UID' not in reader.fieldnames:
            print(f"Warning: 'UID' column not found in {filename}")
            return set()
        return set(row['UID'] for row in reader if row['UID'])


def save_processed_uid(filename, uid):
    ensure_file_exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([uid])


def setup_logger(log_file: str) -> logging.Logger:
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a new logger
    logger = logging.getLogger(f"{os.path.basename(log_file)}")
    logger.setLevel(logging.INFO)

    # Create a file handler for this logger
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


def divide_map(original_map: Dict[str, str], num_threads: int) -> list[Dict[str, str]]:
    # Convert map items to list for easier division
    items = list(original_map.items())
    total_items = len(items)

    # Calculate items per thread (rounded up)
    items_per_thread = math.ceil(total_items / num_threads)

    # Divide the items into sub-maps
    sub_maps = []
    for i in range(0, total_items, items_per_thread):
        sub_map = dict(items[i:i + items_per_thread])
        sub_maps.append(sub_map)

    return sub_maps


def setup_output_directory(output_dir: Path):
    """Create output directory if it doesn't exist"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise Exception(f"Failed to create output directory {output_dir}: {e}")


def cleanup_old_files(output_dir: Path, thread_number: int):
    """Remove old output files if they exist"""
    for i in range(thread_number):
        filename = output_dir / f"id_processed_thread_{i}.txt"
        if filename.exists():
            try:
                filename.unlink()
            except Exception as e:
                print(f"Warning: Could not remove old file {filename}: {e}")

    # Clean up old summary file if it exists
    summary_file = output_dir / "processing_summary.txt"
    if summary_file.exists():
        try:
            summary_file.unlink()
        except Exception as e:
            print(f"Warning: Could not remove old summary file: {e}")


def format_time(seconds: float) -> str:
    """Format time in seconds to hours, minutes, seconds"""
    seconds = seconds if seconds is not None else 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def process_map_with_threads(new_map, old_map, id_map, output_dir, thread_number):
    # Start timing the entire process
    total_start_time = time.time()

    # Read configuration
    # output_dir = Path(config.get_output_directory())
    output_dir_path = Path(output_dir)

    # Setup output directory
    setup_output_directory(output_dir_path)

    # Clean up any old output files
    cleanup_old_files(output_dir_path, thread_number)

    # Divide the map
    sub_maps = divide_map(new_map, thread_number)

    # Create and start threads
    threads = []
    for i in range(len(sub_maps)):
        thread = MapProcessor(i, sub_maps[i], old_map, id_map, output_dir)
        threads.append(thread)
        print(f"\n@@@@@@@@@begin to start thread:  {i}")
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Calculate total execution time
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    # After all threads complete, create a summary file
    summary_file_path = output_dir_path / "processing_summary.txt"
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"Processing completed at: {datetime.now()}\n")
        summary_file.write(f"Total threads used: {len(threads)}\n")
        summary_file.write(f"\nExecution Time Summary:\n")
        summary_file.write("-" * 50 + "\n")
        summary_file.write(
            f"Total execution time: {format_time(total_execution_time)}\n\n")

        # Write execution time for each thread
        summary_file.write("Thread Execution Times:\n")
        for thread in threads:
            summary_file.write(
                f"Thread-{thread.thread_id}: {format_time(thread.execution_time)}\n")

    print("\nExecution Summary:")
    print("-" * 50)
    print(f"Total execution time: {format_time(total_execution_time)}")
    print(
        f"Average time per thread: {format_time(total_execution_time/len(threads))}")
    print(f"Check individual thread output files in: {output_dir}")
    print(f"Check {summary_file_path} for detailed summary")


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

    output_pdf = os.path.join(output_pdf, run_name)
    thread_count = int(config['thread_count'])

    log_directory = config['log_path']
    os.makedirs(log_directory, exist_ok=True)

    log_file1 = 'dup_sop_uids_on_batch1.log'
    log_file2 = 'dup_sop_uids_on_batch2.log'
    log_file3 = 'imgCmpLog.log'
    dup_log_file1 = os.path.join(log_directory, log_file1)
    dup_log_file2 = os.path.join(log_directory, log_file2)
    cmp_log = os.path.join(log_directory, log_file3)

# Start the timer
    start_time = time.time()

    new_map = build_dicom_map(new_folder_path, dup_log_file1)
    old_map = build_dicom_map(old_folder_path, dup_log_file2)
    id_map = read_id_map(id_map_csv_file)

    process_map_with_threads(new_map, old_map, id_map,
                             output_pdf, thread_count)

# Stop the timer
    end_time = time.time()

# Calculate and display the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparing DICOM files based on a configuration file")
    parser.add_argument('config', help="Path to the configuration JSON file")

    args = parser.parse_args()
    main(args.config)
