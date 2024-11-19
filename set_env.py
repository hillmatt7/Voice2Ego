import os

def set_huggingface_token(file_path="huggingface_token.txt"):
    """
    Reads the Hugging Face token from a file and sets it as an environment variable.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The Hugging Face token file '{file_path}' was not found.")
    
    with open(file_path, 'r') as file:
        token = file.read().strip()
    
    # Set the token as an environment variable
    os.environ['HUGGINGFACE_TOKEN'] = token
    print(f"HUGGINGFACE_TOKEN set from {file_path}")

# Run the function
if __name__ == "__main__":
    set_huggingface_token()
