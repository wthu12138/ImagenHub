from imagen_hub.metrics import MetricLPIPS
from imagen_hub.utils import load_image, save_pil_image, get_concat_pil_images
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="LPIPS Metric")
    parser.add_argument("--ref_path",
                        type=str,
                        required=True,
                        help="Path to reference images")
    parser.add_argument("--gen_path",
                        type=str,
                        required=True,
                        help="Path to generated images")
    return parser.parse_args()


def evaluate_all(model, list_real_images, list_generated_images):
    score = [
        model.evaluate(x, y)
        for (x, y) in zip(list_real_images, list_generated_images)
    ]
    print("====> Avg Score: ", sum(score) / len(score))


def get_data_pairs(ref_path, gen_path):
    ref_list = []
    gen_list = []
    for root, dirs, files in os.walk(ref_path):
        for file in files:
            image = load_image(os.path.join(root, file))
            ref_list.append(image)
    for root, dirs, files in os.walk(gen_path):
        for file in files:
            image = load_image(os.path.join(root, file))
            gen_list.append(image)
    return ref_list, gen_list


if __name__ == "__main__":
    args = get_args()
    model = MetricLPIPS()
    ref_list, gen_list = get_data_pairs(args.ref_path, args.gen_path)
    evaluate_all(model, ref_list, gen_list)
