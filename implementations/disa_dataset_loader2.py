import os
import numpy as np

def load_dataset(directory, files_list, log_file=None):
    # If log_file is a string (path), open the file in write mode;
    # otherwise assume it's an open file object.
    file_handle = None
    if log_file is not None:
        if isinstance(log_file, str):
            file_handle = open(log_file, 'w')
        else:
            file_handle = log_file

    # Helper function to print and log messages
    def log(message):
        print(message)
        if file_handle is not None:
            file_handle.write(message + "\n")
            file_handle.flush()
    
    # Initialize an empty list to store the vectors
    dataset = []
    # Initialize an empty list to store the image names
    image_names = []

    # Iterate over every file in the directory
    for filename in files_list:
        # Only read .txt files
        if filename.endswith('.txt'):
            # Open the file for reading
            with open(os.path.join(directory, filename), 'r') as file:
                log(f"Processing file: {filename}")
                
                count_vectors = 0
                # Initialize a variable to keep track of the current line number
                line_number = 1

                # Iterate over each line in the file
                for line in file:
                    # Strip whitespace from the line
                    line = line.strip()

                    # If the line number is odd (vector)
                    if line_number % 2 != 0:
                        # Remove '#' from the start of the image name and append it to the list of image names
                        image_name = line.lstrip('#')
                        image_folder = os.path.splitext(filename)[0]
                        image_names.append(image_folder + '/' + image_name)

                    # If the line number is even (vector)
                    if line_number % 2 == 0:
                        # Parse the vector and append it to the list of vectors
                        vector = [float(value) for value in line.split(',')]
                        dataset.append(vector)
                        count_vectors += 1  # Increment vector count

                    # Increment the line number
                    line_number += 1
                
                log(f"\tNumber of vectors: {count_vectors}")
    
    log("")
    log(f"Number of vectors in all files: {len(dataset)}")
    log(f"Number of dimensions: {len(dataset[0])}")
    
    # Close the file handle if we opened it in this function
    if file_handle is not None and isinstance(log_file, str):
        file_handle.close()
    
    return image_names, dataset