import os
import glob
from disa_dataset_loader import load_dataset


def gather_image_files(root_directory):
    # Dictionary to hold image base name without extension and its full path
    image_files_dict = {}
    top_level_dirs = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    for top_dir in top_level_dirs:
        top_dir_path = os.path.join(root_directory, top_dir)
        all_subdirs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(top_dir_path) for f in dn]

        for subdir in all_subdirs:
            relevant_dirs = ['more relevant results', 'less relevant results']
            if os.path.basename(subdir) in relevant_dirs:
                # Collect all PNG images
                for image_file in glob.glob(os.path.join(subdir, '*.png')):
                    base_name = os.path.splitext(os.path.basename(image_file))[0]
                    image_files_dict[base_name] = image_file

    return image_files_dict


def write_vectors_for_images(root_directory, path_to_disa_vectors):
    image_files_dict = gather_image_files(root_directory)
    files_list = glob.glob(os.path.join(path_to_disa_vectors, '*.txt'))

    # Batch process database files as the database is large
    batch_size = 4
    for i in range(0, len(files_list), batch_size):
        print(i)
        batch_files = files_list[i:i + batch_size]

        # Load the dataset for this batch
        image_names, dataset = load_dataset(root_directory, batch_files)

        # Create dictionary from image names and dataset
        vector_dict = dict(zip(image_names, dataset))

        # Process each image using the dictionary of image paths
        for image_name, image_path in image_files_dict.items():
            vector = vector_dict.get(image_name)
            if vector is not None:
                # Write the vector to a .txt file next to the image
                vector_file_path = os.path.splitext(image_path)[0] + '.txt'
                with open(vector_file_path, 'w') as f:
                    f.write(','.join(map(str, vector)))
                    print(f"Written vector to {vector_file_path}")