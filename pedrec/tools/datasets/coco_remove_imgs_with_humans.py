import json
import os

from tqdm import tqdm


def remove_category(directory: str, img_dir: str, annotation_file_path: str, category: int = 1):
    """
    category 1 == humans
    """

    annotation_file_path = os.path.join(directory, "annotations", annotation_file_path)
    assert os.path.isfile(annotation_file_path)
    with open(annotation_file_path, 'r') as annotation_file:
        posts = json.load(annotation_file)
        images = posts["images"]
        img_dict = {}
        for image in images:
            img_dict[image["id"]] = image
        for annotation in tqdm(posts["annotations"]):
            if annotation["category_id"] == category:
                img_id = annotation["image_id"]
                img = img_dict[img_id]
                img_file = os.path.join(directory, img_dir, img["file_name"])
                if os.path.isfile(img_file):
                    os.remove(img_file)

        a = 1
        # annotations = ijson.items(annotation_file, "annotations")
        #
        # for annotation in annotations:
        #     a = 1


if __name__ == "__main__":
    remove_category("data/datasets/COCO_nohumans/", "val2017", "instances_val2017.json")
    remove_category("data/datasets/COCO_nohumans/", "train2017", "instances_train2017.json")