import pickle
import numpy as np

def get_data_list_and_label(data_df):
    return [(lambda arr: int(arr[2]) - 1)(i[:-1].split(' ')) for i in open(data_df).readlines()]

def fusion_score(global_feature_path, depth_feature_path, hand_feature_path):
    global_branch_data = pickle.load(open(global_feature_path, 'rb'))
    hand_branch_data = pickle.load(open(depth_feature_path, 'rb'))
    depth_branch_data = pickle.load(open(hand_feature_path, 'rb'))
    global_branch_data = np.array(global_branch_data)
    hand_branch_data = np.array(hand_branch_data)
    depth_branch_data = np.array(depth_branch_data)
    result = np.multiply(global_branch_data, hand_branch_data)
    result = np.multiply(result, depth_branch_data)
    result_index = np.argmax(result, axis=1)
    labels = np.array(get_data_list_and_label(args.ground_truth_path))
    acc = (labels == result_index).sum() / len(labels)
    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_feature_path', default="./features_MSA3D_global.pkl")
    parser.add_argument('--depth_feature_path', default="./features_MSA3D_depth.pkl")
    parser.add_argument('--hand_feature_path', default="./features_MSA3D_hand.pkl")
    parser.add_argument('--ground_truth_path', default="/home/chz/IsoGD/test_list.txt")
    args = parser.parse_args()
    acc = fusion_score(args.global_feature_path, args.depth_feature_path, args.hand_feature_path)
    print(acc)