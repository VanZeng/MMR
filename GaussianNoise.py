import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import random
from concurrent.futures import ProcessPoolExecutor


class NoiseController:
    """噪声强度控制器"""

    def __init__(self, config):
        self.strength_factor = config["strength_factor"]
        self.base_sigma = config["base_sigma"]
        self.max_sigma = config["max_sigma"]
        self.gamma = config["gamma"]
        self.variation = config["sigma_variation"]
        self.sigma_range = self.calculate_sigma_range()

    def calculate_sigma_range(self):
        scaled_factor = self.strength_factor ** self.gamma
        main_sigma = self.base_sigma + scaled_factor * (self.max_sigma - self.base_sigma)
        delta = main_sigma * self.variation
        min_sigma = max(1, main_sigma - delta)
        max_sigma = main_sigma + delta
        return (min_sigma, max_sigma)


def add_gaussian_noise(img, sigma_range):
    img_float = img.astype(np.float32) / 255.0
    min_sigma, max_sigma = sigma_range
    sigma = random.uniform(min_sigma / 255.0, max_sigma / 255.0)
    noise = np.random.normal(loc=0.0, scale=sigma, size=img_float.shape)
    noisy_img = np.clip(img_float + noise, 0.0, 1.0)
    return (noisy_img * 255).astype(np.uint8), sigma * 255


def process_image(args):
    src_path, dst_path, sigma_range, png_compression = args
    try:
        img = cv2.imread(src_path)
        if img is None:
            print(f"无法读取图像: {src_path}")
            return
        noisy_img, _ = add_gaussian_noise(img, sigma_range)  # 不再返回sigma值
        cv2.imwrite(dst_path, noisy_img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    except Exception as e:
        print(f"处理 {src_path} 时发生错误: {str(e)}")


def process_dataset(original_root, output_root, controller, num_workers, png_compression):
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    file_pairs = []
    for root, _, files in os.walk(original_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, original_root)
                dst_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '.png')
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                file_pairs.append((src_path, dst_path, controller.sigma_range, png_compression))

    # 仅处理不收集结果
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_image, file_pairs),
                  total=len(file_pairs),
                  desc="处理进度"))


def generate_examples(original_root, example_dir, controller, png_compression, num_examples):
    os.makedirs(example_dir, exist_ok=True)
    image_paths = []
    for root, _, files in os.walk(original_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    selected = random.sample(image_paths, min(num_examples, len(image_paths)))
    for idx, src_path in enumerate(tqdm(selected, desc="生成示例")):
        try:
            img = cv2.imread(src_path)
            if img is None:
                continue
            noisy_img, actual_sigma = add_gaussian_noise(img, controller.sigma_range)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Original", (10, 30), font, 0.8, (0, 255, 0), 2)
            info_text = [
                f"Strength Factor: {controller.strength_factor:.2f}",
                f"Actual Sigma: {actual_sigma:.1f}",
                f"Theory Range: {controller.sigma_range[0]:.1f}-{controller.sigma_range[1]:.1f}"
            ]
            y_start = 40
            for line in info_text:
                cv2.putText(noisy_img, line, (10, y_start), font, 0.7, (0, 255, 0), 2)
                y_start += 30
            resized_orig = cv2.resize(img, (512, 512))
            resized_noisy = cv2.resize(noisy_img, (512, 512))
            comparison = cv2.hconcat([resized_orig, resized_noisy])
            filename = f"example_{idx + 1}_{os.path.basename(src_path).split('.')[0]}.png"
            cv2.imwrite(os.path.join(example_dir, filename), comparison,
                        [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        except Exception as e:
            print(f"生成示例失败: {src_path} - {str(e)}")


def apply_noise(
        strength_factor: float,
        input_image_path: str,
        output_image_path: str,
        example_image_path: str,
        base_sigma: int = 15,
        max_sigma: int = 120,
        sigma_variation: float = 0.15,
        gamma: float = 1.5,
        random_seed: int = 42,
        num_examples: int = 10,
        num_workers: int = 8,
        png_compression: int = 0
):
    config = {
        "strength_factor": strength_factor,
        "base_sigma": base_sigma,
        "max_sigma": max_sigma,
        "sigma_variation": sigma_variation,
        "gamma": gamma,
        "random_seed": random_seed,
        "num_examples": num_examples,
        "num_workers": num_workers,
        "png_compression": png_compression
    }
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    controller = NoiseController(config)
    print("\n噪声参数配置:")
    print(f"- 综合因子: {config['strength_factor']:.2f}")
    print(f"- 理论范围: {controller.sigma_range[0]:.1f} ~ {controller.sigma_range[1]:.1f}")
    print(f"- 波动系数: ±{config['sigma_variation'] * 100:.0f}%")
    print(f"- 非线性系数: {config['gamma']:.1f}")
    process_dataset(
        original_root=input_image_path,
        output_root=output_image_path,
        controller=controller,
        num_workers=config["num_workers"],
        png_compression=config["png_compression"]
    )
    generate_examples(
        original_root=input_image_path,
        example_dir=example_image_path,
        controller=controller,
        png_compression=config["png_compression"],
        num_examples=config["num_examples"]
    )
    print("\n处理完成！")
    print(f"输出目录: {os.path.abspath(output_image_path)}")
    print(f"示例目录: {os.path.abspath(example_image_path)}")