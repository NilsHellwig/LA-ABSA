import os
import shutil
from send2trash import send2trash

class DotDict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")
        
    

def clean_up(output_directory='./classifier/outputs'):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
        print(f'Directory {output_directory} has been deleted.')
    else:
        print(f'Directory {output_directory} does not exist.')

    example_file_path = 'example.txt'  # Beispiel-Datei, die existieren muss

    if os.path.exists(example_file_path):
        send2trash(example_file_path)
        print(f'File {example_file_path} has been moved to trash.')
    else:
        print(f'File {example_file_path} does not exist; nothing to move to trash.')



def create_output_directory(output_directory='./classifier/outputs'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f'Directory {output_directory} has been created.')
    else:
        print(f'Directory {output_directory} already exists.')
