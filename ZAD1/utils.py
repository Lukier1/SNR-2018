import os

def output_root(filter_size, filter_depth):
    directory = f"images/output{filter_size}X{filter_size}_{filter_depth}bit" 
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory

def model_output_filename(filter_size, filter_depth, layer_num):
    directory = "models" 
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory + f"/model_{filter_size}x{filter_size}_{filter_depth}bit_{layer_num}layers"