import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.stats import mode

# CSV file의 range 정하기
file_numbers = range(501, 519)  # 501 ~ 518
all_sequences = []

# 각 csv 처리하기
for num in file_numbers:
    filename = f'{num}.csv'
    print(f"Processing {filename}...")
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # pandas 내장 시간함수
    df['second'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # 년도-월-일 시간:분:초 까지만 추출

    # Sensor들의 mean 구하기
    sensor_means = df.groupby('second').mean().reset_index()

    # label 지정
    all_labels = [1, 3, 4, 5, 6, 7, 8]

    # label distribution 계산. 이 때 unstack를 통해 등장하지 않은 label에도 0 할당
    label_distributions = df.groupby('second')['label'].apply(lambda x: x.value_counts(normalize=True)).unstack(fill_value=0).reindex(columns=all_labels, fill_value=0)

    # Merge
    final_df = pd.merge(sensor_means, label_distributions, on='second', how='left')
    
    # 30초씩 묶기
    num_rows = len(final_df)
    if num_rows >= 30:  # 30초보다 데이터 수가 적은 경우는 포함 X
        for start in range(0, num_rows - num_rows % 30, 30):
            segment = final_df.iloc[start:start + 30]
            features = segment[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values.flatten()  # flatten을 통해 1차원 배열로 변환 -> 하나의 feature vector
            
            # label 묶기 - 가장 많이 나타나는 것으로 
            labels = segment[all_labels].values
            label_counts = np.argmax(labels, axis=1)
            most_common_label = mode(label_counts)[0]
            
            one_hot_label = np.zeros(len(all_labels))
            one_hot_label[most_common_label] = 1
            
            all_sequences.append((features, one_hot_label))

# Feature와 label 분리
features, labels = zip(*all_sequences)
X = np.array(features)
Y = np.array(labels)

# NumPy 배열을 텐서로 변환
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# 7:3의 비율로 train / test 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 저장
np.savetxt('training_set_X.txt', X_train, fmt='%f')
np.savetxt('test_set_X.txt', X_test, fmt='%f')
torch.save(Y_train, 'training_set_Y.tensor')
torch.save(Y_test, 'test_set_Y.tensor')
