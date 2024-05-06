import json
from imagen_hub.utils import load_image
import os
from imagen_hub.metrics import MetricCLIPScore
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="CLIPScore Metric")
    parser.add_argument(
        "--lookup_path",
        type=str,
        required=True,
        help="Path to the lookup file containing image-text pairs",
    )
    parser.add_argument(
        "--gen_path",
        type=str,
        required=True,
        help="Path to the generated images",
    )
    return parser.parse_args()


def get_data_pairs(loopup_path, gen_path):
    gen_image_list = []
    prompt_list = []
    with open(loopup_path, 'r') as file:
        data = json.load(file)
    file.close()
    for key, value in data.items():
        for root, _, files in os.walk(gen_path):
            if key in files:
                image = load_image(os.path.join(root, key))
                gen_image_list.append(image)
                prompt_list.append(value['target_global_caption'])
    return gen_image_list, prompt_list


def evaluate_all(model, list_gen_images, list_prompts):
    score = [
        model.evaluate(x, y) for (x, y) in zip(list_gen_images, list_prompts)
    ]
    print("====> Avg Score: ", sum(score) / len(score))


if __name__ == '__main__':
    args = get_args()
    model = MetricCLIPScore()
    gen_image_list, prompt_list = get_data_pairs(args.lookup_path,
                                                 args.gen_path)
    evaluate_all(model, gen_image_list, prompt_list)
