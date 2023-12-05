import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import glob

# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # 현재 작업 디렉토리 내의 파일 목록 얻기
    file_list = os.listdir("/opt/ml/input/data/training")
    print(file_list)

    # 결과 출력
    print("현재 작업 디렉토리의 파일 및 디렉토리 목록:")
    for file in file_list:
        print(file)

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n_neighbors", type=int, default=5)
    
    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--user_id", type=str, default="AIGNER-Debut")

    args, _ = parser.parse_known_args()

    # default는 sagemaker에 들어있는 .npy 경로
    train_x_data = np.load('/opt/ml/input/data/training/train_x.npy', allow_pickle=True)
    train_y_data = np.load('/opt/ml/input/data/training/train_y.npy', allow_pickle=True)
    dataset_label = np.load('/opt/ml/input/data/training/dataset_columns.npy', allow_pickle=True)
    
    
    train_x_data = pd.DataFrame(train_x_data, columns=dataset_label.tolist())
    print(train_x_data.shape)
    train_y_data = pd.DataFrame({"type_encoded": train_y_data.tolist()})
    print(train_y_data)
    
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors)
    model.fit(train_x_data, train_y_data["type_encoded"])
    
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)