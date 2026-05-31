import os
import numpy as np

def load_dataset(directory, files_list):
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
                print(f"Processing file: {filename}")
                
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
                
                print(f"\tNumber of vectors: {count_vectors}")
    
    print()
    print(f"Number of vectors in all files: {len(dataset)}")
    print(f"Number of dimensions: {len(dataset[0])}")
    
    return image_names, dataset