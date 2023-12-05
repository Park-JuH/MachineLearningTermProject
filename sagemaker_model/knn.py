import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Dataset Overview - buyer dataset
buyer_dataset = pd.read_csv("./noon_perfumes_buyer_dataset.csv")
print(f"Rows: {buyer_dataset.shape[0]}\nColumns: {buyer_dataset.shape[1]}")
# Using with to print all columns
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_dataset.head())

buyer_dataset.info()

buyer_dataset.describe()

buyer_dataset.describe(exclude = np.number)

buyer_dataset.isnull().sum()


# 1. Dataset Overview - perfumes dataset
noon_perfumes = pd.read_csv("./noon_perfumes_dataset.csv")
print(f"Rows: {noon_perfumes.shape[0]}\nColumns: {noon_perfumes.shape[1]}")
# Using with to print all columns
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(noon_perfumes.head())

noon_perfumes.info()

noon_perfumes.describe()

noon_perfumes.describe(exclude = np.number)

noon_perfumes.isnull().sum()


# 2.Data Preprocessing - buyer dataset

# [dirty data] drop NaN data
buyer_dataset = buyer_dataset.dropna(axis=0)


# [dirty data] Delete null string data
buyer_dataset = buyer_dataset[buyer_dataset['preference_base_note'] != '']
buyer_dataset = buyer_dataset[buyer_dataset['preference_middle_note'] != '']


# [Data cleaning] Integrate brand and name into one feature
buyer_dataset['type'] = buyer_dataset['brand'] + '-' + buyer_dataset['name']
buyer_dataset = buyer_dataset.drop(['brand', 'name'], axis=1)

# Create SVD's pivot table
usr_pf_table = buyer_dataset.pivot_table(index='user_id', columns='type', values='satisfaction', aggfunc='mean').fillna(0)
pivot_table = usr_pf_table.values.T
pivot_true = buyer_dataset['satisfaction']

buyer_dataset = buyer_dataset.drop(['user_id'], axis=1)
buyer_dataset = buyer_dataset.reset_index(drop=True)

#Copy the original dataset
org_buyer_dataset = buyer_dataset

# [one-hot encoding] preference_base_note & preference_middle_note
def get_buyer_notes_type_set(dataset):
    note_set = set()
    for i in range(dataset.shape[0]):
        note_set.add(buyer_dataset.loc[i]['preference_base_note'])
        note_set.add(buyer_dataset.loc[i]['preference_middle_note'])
    return note_set

def change_categorical_notes_to_encoding(dataset):
    for i in range(dataset.shape[0]):
        dataset.at[i, dataset.loc[i]['preference_base_note']] = 1
        dataset.at[i, dataset.loc[i]['preference_middle_note']] = 1
    dataset = dataset.drop(['preference_base_note'], axis=1)
    dataset = dataset.drop(['preference_middle_note'], axis=1)
    return dataset

note_set = get_buyer_notes_type_set(buyer_dataset)

for note in note_set:
    buyer_dataset = pd.concat([buyer_dataset, pd.DataFrame({note: [0 for i in range(buyer_dataset.shape[0])]})], axis=1)

buyer_dataset_pre=change_categorical_notes_to_encoding(buyer_dataset)

buyer_copy = buyer_dataset_pre
# buyer_original = buyer_dataset

# Initialize Label Encoder
label_encoder_gd = LabelEncoder()
label_encoder_type = LabelEncoder()
buyer_copy['buyer_gender'] = label_encoder_gd.fit_transform(buyer_copy['buyer_gender'])
buyer_copy['type'] = label_encoder_type.fit_transform(buyer_dataset['type'])

# Print buyer dataset
print("BUYER_DATASET")
print(f"Rows: {buyer_dataset.shape[0]}\nColumns: {buyer_dataset.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_dataset.head())

# Print Buyer dataset_copy
print("buyer_copy")
print(f"Rows: {buyer_copy.shape[0]}\nColumns: {buyer_copy.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_copy.head())


# 2.Data Preprocessing - perfume dataset

# [dirty data] drop useless column
noon_perfumes=noon_perfumes.drop(['Unnamed: 0'], axis=1)
noon_perfumes=noon_perfumes.drop(['seller'], axis=1)


# [dirty data] Delete null string data
noon_perfumes = noon_perfumes[noon_perfumes['base_note'] != '']
noon_perfumes = noon_perfumes[noon_perfumes['middle_note'] != '']


# [Data cleaning] Integrate brand and name into one feature
brand_dict={'Dorall':'DORALL COLLECTION',
            'Al Rasasi':'Rasasi',
            'YVES':'Yves Saint Laurent',
            'YSL':'Yves Saint Laurent',
            'Benetton':'UNITED COLORS OF BENETTON',
            'LANVIN PARIS':'LANVIN',
            'Genie':'Genie Collection',
            'Justcavalli':'Roberto Cavalli',
            'Mont Blanc':'MONTBLANC',
            'Parfums Gres':'Gres',
            'EMPORIO ARMANI':'GIORGIO ARMANI' ,
            'ST Dupont':'S.T.Dupont' ,
            'Dupont':'S.T.Dupont',
            'Roberto':'Roberto Cavalli',
            'marbert man':'Marbert'}
noon_perfumes=noon_perfumes.replace({"brand": brand_dict})

noon_perfumes['middle_note'] = noon_perfumes['middle_note'].str.replace(' And',',')
noon_perfumes['middle_note'] = noon_perfumes['middle_note'].str.replace(' and',',')
noon_perfumes['middle_note'] = noon_perfumes['middle_note'].str.replace(" ","")

noon_perfumes['base_note'] = noon_perfumes['base_note'].str.replace(' And',',')
noon_perfumes['base_note'] = noon_perfumes['base_note'].str.replace(' and',',')
noon_perfumes['base_note'] = noon_perfumes['base_note'].str.replace(" ","")

noon_perfumes['scents'] = noon_perfumes['scents'].str.replace(' And',',')
noon_perfumes['scents'] = noon_perfumes['scents'].str.replace(' and',',')
noon_perfumes['scents'] = noon_perfumes['scents'].str.replace(" ","")

noon_perfumes['type'] = noon_perfumes['brand'] + '-' + noon_perfumes['name']
noon_perfumes = noon_perfumes.drop(['brand', 'name'], axis=1)


# [one-hot encoding] concentration
noon_perfumes = pd.get_dummies(noon_perfumes, columns=['concentration'])


# [one-hot encoding] scents & base_note & middle_note
def get_notes_type_set(dataset, type, note_set):
    for i in range(dataset.shape[0]):
        for j in dataset.loc[i][type].split(','):
            note_set.add(j)

def change_categorical_notes_to_encoding(dataset, type):
    for i in range(dataset.shape[0]):
        for j in dataset.loc[i][type].split(','):
            dataset.at[i, j] = 1


perfume_note_set = set()

for type in ['base_note', 'middle_note', 'scents']:
    get_notes_type_set(noon_perfumes, type, perfume_note_set)

for note in perfume_note_set:
    noon_perfumes = pd.concat([noon_perfumes, pd.DataFrame({note: [0 for i in range(noon_perfumes.shape[0])]})], axis=1)

for type in ['base_note', 'middle_note', 'scents']:
    change_categorical_notes_to_encoding(noon_perfumes, type)
    noon_perfumes = noon_perfumes.drop([type], axis=1)



print("noon_perfumes")
print(f"Rows: {noon_perfumes.shape[0]}\nColumns: {noon_perfumes.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(noon_perfumes.head())


# 2.Data Preprocessing - create buyer matrix
user_index_set = set(buyer_dataset['user_id'])
buyer_matrix = dict()
buyer_matrix.update({idx: [0 for i in range(len(note_set))] for idx in user_index_set if idx not in buyer_matrix})

for i in range(buyer_dataset.shape[0]):
  user_id = buyer_dataset.loc[i]['user_id']
  for idx, note in enumerate(note_set):
    buyer_matrix[user_id][idx] += (buyer_dataset.loc[i][note] * buyer_dataset.loc[i]['satisfaction'])
buyer_dataFrame = pd.DataFrame(buyer_matrix, index=list(note_set))
buyer_dataFrame = buyer_dataFrame.T
buyer_dataFrame
buyer_dataFrame.to_csv('./test.csv', index=True)

print(f"Rows: {buyer_dataFrame.shape[0]}\nColumns: {buyer_dataFrame.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_dataFrame.head())

# Copy original
buyer_dataset_original = buyer_dataset.copy()
noon_perfumes_original = noon_perfumes.copy()

# Collaborative filtering perfume recommendation
# After merging data, create user-item metrics and set up knn model
# Recommend other perfumes that users may prefer for a specific perfume
merged_data = pd.merge(buyer_dataset, noon_perfumes, on='type', how='inner')

# Pivot the data to create a user-item matrix
user_item_matrix = merged_data.pivot_table(index='user_id', columns='type', values='satisfaction')
user_item_matrix = user_item_matrix.fillna(0)

# Transpose the matrix for item-based collaborative filtering
item_user_matrix = user_item_matrix.T

# KNN model for item-based collaborative filtering
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(item_user_matrix)

# Function to make recommendations based on a given perfume
def make_recommendations(perfume_name, data, model, n_recommendations=5):
    if perfume_name not in data.index:
        return f"Perfume '{perfume_name}' not found in dataset."

    perfume_idx = data.index.tolist().index(perfume_name)
    distances, indices = model.kneighbors(data.iloc[perfume_idx, :].values.reshape(1, -1), n_neighbors=n_recommendations+1)

    closest_perfumes = [data.index[i] for i in indices.flatten()[1:]]  # Skip the first one
    distances_to_perfumes = distances.flatten()[1:]

    
    recommendations = dict(zip(closest_perfumes, distances_to_perfumes))
    return recommendations

# Example usage
example_perfume = item_user_matrix.index[0]
recommendations = make_recommendations(example_perfume, item_user_matrix, model_knn)
print(recommendations)

item_user_matrix


# When user 1 wants to be recommended another perfume with similar characteristics to 'ADOLFO DOMINGUEZ-Vetiver Hombre'
# Example usage
example_perfume = item_user_matrix.index[0]
recommendations = make_recommendations(example_perfume, item_user_matrix, model_knn)

# Formatting the output
print(f"Recommendations for '{example_perfume}':\n")
for perfume, distance in recommendations.items():
    print(f"{perfume}: Similarity Distance = {distance:.3f}")


#Recommended Predictions
def train_evaluate_knn(data, target_variable, n_neighbors, test_size=0.2, random_state=42):

    # Target Settings
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)

    # knn 
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Training
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    return accuracy, classification_report_str


def plot_knn_elbow_graph(data,target_variable, neighbors_range): # Find the optimal k value
    error_rates = []
    # Target 
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    for n_neighbors in neighbors_range:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rates.append(np.mean(pred_i != y_test))

    # Elobw graph
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors_range, error_rates, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Error Rate')
    plt.show()

# buyer_dataset_original 
buyer_dataset_original = buyer_dataset_original.drop(['type'], axis=1)
buyer_dataset_original.head()

# Training and evaluating KNN with k=3
accuracy_k3, classification_report_k3 = train_evaluate_knn(buyer_dataset_original, 'satisfaction', 3)
print(f"KNN Model Accuracy(k=3): {accuracy_k3}")
print(classification_report_k3)

neighbors_range = range(1, 20)

# Find the value of k
#k=3 knn model performance 34
#Finding optimal k value with Elbow mehtod -> 16
neighbors_range = range(1, 20)
plot_knn_elbow_graph(buyer_dataset_original,'satisfaction', neighbors_range)

test_dataset = test_dataset.drop(['Unnamed: 0'], axis=1)

print("test datset")
print(f"Rows: {test_dataset.shape[0]}\nColumns: {test_dataset.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(test_dataset.head())
print()
print("buyer datset")   
print(f"Rows: {buyer_dataset.shape[0]}\nColumns: {buyer_dataset.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_dataset.head())
    
print()
print("original buyer datset")   
print(f"Rows: {buyer_dataset_original.shape[0]}\nColumns: {buyer_dataset_original.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_dataset_original.head())


buyer_dataset.loc[buyer_dataset['user_id']==1].values

buyer_dataset_copy = buyer_dataset.copy()


# Target, feature separation
X = buyer_dataset.drop(['type_encoded', 'type', 'user_id'], axis=1)
y = buyer_dataset['type_encoded'].astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model creation
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)

# Predict
y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"KNN Model Accuracy: {accuracy}")


# Check the train column
print(X_train.columns.tolist())

# Selecting a few continuous columns for outlier visualization
selected_columns = ['satisfaction', 'buyer_age', 'user_id']

# Plotting boxplots for the selected columns
plt.figure(figsize=(12, 6))
for i, col in enumerate(selected_columns):
    plt.subplot(1, len(selected_columns), i + 1)
    sns.boxplot(y=buyer_dataset[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Before applying MinMax
accuracy, classification_report_result = train_evaluate_knn(buyer_dataset_original, 'satisfaction',16)
print(f"KNN Model Accuracy(k=16): {accuracy}")
print(classification_report_result)


# Save original
original_buyer_age = buyer_dataset['buyer_age'].copy()
original_user_id = buyer_dataset['user_id'].copy()

buyer_dataset_minmax = buyer_dataset_original.copy()

# minMax
minmax_scaler = MinMaxScaler()

# Applying MinMaxScaler 'buyer_age' and 'user_id' (Excluding target)
buyer_dataset_minmax[['buyer_age', 'user_id']] = minmax_scaler.fit_transform(buyer_dataset_minmax[['buyer_age', 'user_id']])
accuracy, classification_report_result = train_evaluate_knn(buyer_dataset_minmax, 'satisfaction',16)
print(f"KNN Model Accuracy(buyer_age, user_id - MinMax): {accuracy}")
print(classification_report_result)


# Apply satisfacction minmax
# Identify continuous columns (excluding binary columns)
continuous_columns = ['satisfaction', 'buyer_age', 'user_id']

# Initializing the MinMaxScaler
minmax_scaler = MinMaxScaler()

# Applying the MinMaxScaler to the data
buyer_data_minmax_scaled = buyer_dataset_original.copy()
buyer_data_minmax_scaled[continuous_columns] = minmax_scaler.fit_transform(buyer_data_minmax_scaled[continuous_columns])

# Defining the target variable and features for the minmax scaled data
X_minmax = buyer_data_minmax_scaled.drop(['satisfaction'], axis=1)
y_minmax = buyer_data_minmax_scaled['satisfaction'].astype(int)

# Splitting the data into training and testing sets for the minmax scaled data
X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = train_test_split(X_minmax, y_minmax, test_size=0.2, random_state=42)

# Initializing and fitting the KNN classifier to the minmax scaled training data
knn_minmax = KNeighborsClassifier(n_neighbors=16)
knn_minmax.fit(X_train_minmax, y_train_minmax)

# Predicting on the minmax scaled test data
y_pred_minmax = knn_minmax.predict(X_test_minmax)

# Evaluating the model for the minmax scaled data
accuracy_minmax = accuracy_score(y_test_minmax, y_pred_minmax)
classification_rep_minmax = classification_report(y_test_minmax, y_pred_minmax)


print(f"KNN Model Accuracy: {accuracy_minmax}")
print(classification_report_result)


# Recommended perfumes with the highest satisfaction
# Sort satisfaction scores in descending order, and get the sorted index.
sorted_indices = np.argsort(y_test_minmax)[::-1]

# Set N to recommend the top N perfumes.
N = 5  # Top 5 perfumes recommended as examples

# Get information on top N perfumes.
recommended_perfumes = X_test_minmax.iloc[sorted_indices[:N]]
recommended_perfumes['type'] = recommended_perfumes['type_encoded'].map(lambda x: next((name for name, code in type_encoding_dict.items() if code == x), None))
for index in recommended_perfumes.index:
    recommended_perfumes.at[index, 'user_id'] = buyer_dataset.loc[index, 'user_id']
    recommended_perfumes.at[index, 'buyer_age'] = buyer_dataset.loc[index, 'buyer_age']


# Print information on recommended perfumes
print("추천 향수 정보:")
recommended_perfumes

# Select the columns we need
selected_columns = ['user_id', 'buyer_gender', 'buyer_age', 'type']

# Filter only those one-hot encoded columns with a value of 1
for index, row in recommended_perfumes.iterrows():
    one_hot_encoded_columns = row[row == 1].index.tolist()
    selected_columns.extend(one_hot_encoded_columns)

# Deduplication
selected_columns = list(set(selected_columns))

# Filtered result output
recommended_perfumes[selected_columns]


## Data used in train learning

print(f"Rows: {X_train_minmax.shape[0]}\nColumns: {X_train_minmax.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X_train_minmax.head())

print()
print("X_train_minmax")
print(f"Rows: {X_train_minmax.shape[0]}\nColumns: {X_train_minmax.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X_train_minmax.head())