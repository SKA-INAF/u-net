from pathlib import Path
import argparse
import json
from pathlib import Path
import shutil

def main(args):

    test_dir = Path('data/test')
    testannot_dir = Path('data/testannot')

    snr_lt_5_folder_img = Path('snr_data/lt_5/test')
    snr_lt_5_folder_annot = Path('snr_data/lt_5/testannot')
    snr_5_to_20_folder_img = Path('snr_data/5_to_20/test')
    snr_5_to_20_folder_annot = Path('snr_data/5_to_20/testannot')
    snr_gt_20_folder_img = Path('snr_data/gt_20/test')
    snr_gt_20_folder_annot = Path('snr_data/gt_20/testannot')

    snr_lt_5_folder_img.mkdir(parents=True, exist_ok=True)
    snr_lt_5_folder_annot.mkdir(parents=True, exist_ok=True)
    snr_5_to_20_folder_img.mkdir(parents=True, exist_ok=True)
    snr_5_to_20_folder_annot.mkdir(parents=True, exist_ok=True)
    snr_gt_20_folder_img.mkdir(parents=True, exist_ok=True)
    snr_gt_20_folder_annot.mkdir(parents=True, exist_ok=True)

    test_paths = list(test_dir.glob('*.png'))
    test_paths = [p.stem for p in test_paths]

    snr_entry = []
    with open('test_snr_split.txt') as snr:
        for line in snr:
            img_name, value = line.strip().split()
            value = float(value)
            if img_name not in test_paths:
                continue
            img_path = test_dir / Path(img_name).with_suffix('.png')
            annot_path = testannot_dir / Path(img_name).with_suffix('.png')
            if value <= 5:
                shutil.copyfile(img_path, snr_lt_5_folder_img / Path(img_name).with_suffix('.png'))
                shutil.copyfile(annot_path, snr_lt_5_folder_annot / Path(img_name).with_suffix('.png'))
            if value > 5 and value <= 20:
                shutil.copyfile(img_path, snr_5_to_20_folder_img / Path(img_name).with_suffix('.png'))
                shutil.copyfile(annot_path, snr_5_to_20_folder_annot / Path(img_name).with_suffix('.png'))
            if value > 20:
                shutil.copyfile(img_path, snr_gt_20_folder_img / Path(img_name).with_suffix('.png'))
                shutil.copyfile(annot_path, snr_gt_20_folder_annot / Path(img_name).with_suffix('.png'))
            # snr_entry.append(line)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_dir", default="sample_jsons", help="Directory of JSON files")
    parser.add_argument("--gt_json", default="gt_boxes.json", help="JSON file with ground truth annotations")
    parser.add_argument("--pred_json", default="pred_boxes.json", help="JSON file with prediction annotations")
    parser.add_argument("--split_folder", default="snr_splits", help="Folder with SNR based split")

    # Optional argument flag which defaults to False

    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)

    main(args)