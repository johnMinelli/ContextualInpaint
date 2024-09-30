import json
from collections import Counter
import matplotlib.pyplot as plt


def read_json(files_path):
    json_data = []
    for file_path in files_path:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
        json_data.extend(file_data)
    return json_data


def get_valid_triplets(synth_data):
    i = 0 
    valid_triplets = set()
    for item in synth_data:
        for annotation in item['hoi_annotation']:
            subject_category = item["annotations"][annotation['subject_id']]['category_id']
            object_category = item["annotations"][annotation['object_id']]['category_id']
            role_category = annotation['category_id']
            if annotation['object_id'] >= 0 and object_category > 1:
                i += 1
                valid_triplets.add((subject_category, role_category, object_category))
    return i
    return valid_triplets


def count_triplets(data, objects_filter=True):
    triplet_counter = Counter()

    for item in data:
        elements = item['annotations']
        for annotation in item['hoi_annotation']:
            subject_id = annotation['subject_id']
            object_id = annotation['object_id']

            subject_category = elements[subject_id]['category_id']
            role_category = annotation['category_id']
            object_category = elements[object_id]['category_id']
            if not objects_filter or (object_id >= 0 and object_category > 1):
                triplet = (subject_category, role_category, object_category)
                triplet_counter[triplet] += 1

    return triplet_counter


def plot_triplets(triplet_counter):
    sorted_triplets = sorted(triplet_counter.items(), key=lambda x: x[1], reverse=True)
    triplets, counts = zip(*sorted_triplets)

    plt.figure(figsize=(15, 8))
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Triplets (Sorted by Frequency)')
    plt.ylabel('Number of Occurrences (Log Scale)')
    plt.title('Triplet Occurrences in HOI Annotations')
    plt.yscale('log')
    plt.xticks([])
    plt.tight_layout()
    plt.savefig('triplet_occurrences.png')
    plt.show()
    plt.close()


def main(files_path):
    data = read_json(files_path)
    triplet_counter = count_triplets(data)
    plot_triplets(triplet_counter)
    print(f"Total unique triplets: {len(triplet_counter)}")
    print("Top 10 most common triplets:")
    for triplet, count in triplet_counter.most_common(10):
        print(f"{triplet}: {count}")


if __name__ == "__main__":
    real_annotations = "/data01/gio/ctrl/data/vcoco/annotations/trainval_vcoco.json"
    synth_annotations = "/data01/gio/ctrl/out/train_data_gen_th_vcoco_cg25/gen_fin.json"
    main([real_annotations])