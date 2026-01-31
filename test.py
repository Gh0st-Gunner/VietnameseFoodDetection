# Simple script to count images in each category for train, val, and test splits

import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd


def count_images_in_categories(base_dir):
    data = []
    total = 0
    if not os.path.isdir(base_dir):
        return data, total
    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            image_count = sum(
                1 for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))
            )
            data.append([category, image_count])
            total += image_count
    return data, total


if __name__ == "__main__":
    all_tables = {}
    splits = ['train', 'val', 'test']
    for split in splits:
        data, total = count_images_in_categories(os.path.join('vnfood_combined_dataset', split))
        all_tables[split] = (data, total)

    # Print tables in terminal
    for split in splits:
        data, total = all_tables[split]
        print(f"\n{split.upper()} split:")
        print(tabulate(data, headers=["Category", "Image Count"]))
        print(f"Total images in {split}: {total}")

    # Save bar chart, pie chart, and txt summary in data_overview folder
    os.makedirs('data_overview', exist_ok=True)
    for split in splits:
        data, total = all_tables[split]
        if not data:
            continue
        df = pd.DataFrame(data, columns=["Category", "Image Count"])
        # Sort by image count descending for better visualization
        df = df.sort_values(by="Image Count", ascending=False)
        # Bar chart
        plt.figure(figsize=(max(10, min(0.3*len(df), 30)), 8))
        plt.bar(df["Category"], df["Image Count"], color='skyblue')
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Image Count", fontsize=12)
        plt.title(f"{split.upper()} split - Image Count per Category", fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"data_overview/{split}_image_count_chart.png", bbox_inches='tight')
        plt.close()
        # Pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(df["Image Count"], labels=df["Category"], autopct='%1.1f%%')
        plt.title(f"{split.upper()} split - Image Count Distribution")
        plt.tight_layout()
        plt.savefig(f"data_overview/{split}_image_count_pie.png", bbox_inches='tight')
        plt.close()
        # Txt summary
        with open(f"data_overview/{split}_image_count_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"{split.upper()} split - Image Count per Category\n")
            f.write(df.to_string(index=False))
            f.write(f"\nTotal images in {split}: {total}\n")

    # Create overall dataset summary
    with open("data_overview/dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("Dataset Summary\n")
        f.write("=" * 50 + "\n")
        total_images = 0
        for split in splits:
            data, total = all_tables[split]
            f.write(f"{split.upper()} split: {total} images\n")
            total_images += total
        f.write(f"\nTotal images across all splits: {total_images}\n")
        # Assuming all splits have same categories
        if all_tables['train'][0]:
            num_categories = len(all_tables['train'][0])
            f.write(f"Number of categories: {num_categories}\n")
            f.write("Categories: " + ", ".join([row[0] for row in all_tables['train'][0]]) + "\n")
        f.write("\nGenerated on: " + str(pd.Timestamp.now()) + "\n")

def count_images_in_categories(base_dir):
    if not os.path.isdir(base_dir):
        print(f"{base_dir} does not exist.")
        return
    print(f"\nCounting images in: {base_dir}")
    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            image_count = sum(
                1 for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))
            )
            print(f"{category}: {image_count} images")

for split in ['train', 'val', 'test']:
    count_images_in_categories(os.path.join('vnfood_combined_dataset', split))