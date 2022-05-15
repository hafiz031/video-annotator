import argparse
import datetime
from annotationutils.annotator import Annotator
from annotationutils.utils import generate_timestamp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Welcome to Video-Annotator!")
    parser.add_argument("-d", "--file_dir", 
                        action = "store_true", 
                        help = "The directory of the video file.")
    parser.add_argument("-o", "--output_dir", 
                        action = "store_true", 
                        default = f"outputs/{generate_timestamp()}"
                        help = "The output directory of annotation and corresponding frames.")
    args = parser.parse_args()

    annotator = Annotator(file_dir = args.file_dir,
            output_dir = args.output_dir)

    annotator.do_annotation()