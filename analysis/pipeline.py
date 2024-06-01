import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
import faiss
from collections import Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report


def save_faiss_index(faiss_index, filename):
    faiss.write_index(faiss_index, filename)

def unzip_folder(zip_file_path, extract_to_directory):
    # Ensure the target directory exists
    if not os.path.exists(extract_to_directory):
        os.makedirs(extract_to_directory)
    
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(extract_to_directory)

def write_texts_to_csv(root_dir, output_csv):
    # List to store data
    data = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                filepath = os.path.join(dirpath, filename)
                # Read the content of the text file
                with open(filepath, 'r', encoding='utf-8') as file:
                    contents = file.read()
                    # Extract subdirectory name
                    subdirectory = os.path.basename(dirpath)
                    # Append to the list
                    data.append({'filename': filename, 'class': subdirectory, 'contents': contents})
    df = pd.DataFrame(data)

    # Write DataFrame to CSV
    df.to_csv(output_csv, index=False)

async def load_and_split_data(csv):
    data = pd.read_csv(csv)
    embeddings = await get_embeddings_async(data)
    # Concatenate original DataFrame with embeddings DataFrame
    result_df = pd.concat([data, embeddings], axis=1)
    train_data, test_data = train_test_split(result_df, test_size=0.2, stratify=data['class'], random_state=42)
    train_data['set'] = 'train'
    test_data['set'] = 'test'
    return train_data, test_data

async def get_embeddings_async(data, model_name='bert-base-uncased'):
    # Load tokenizer and model for TensorFlow
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    
    loop = asyncio.get_running_loop()

    # Define a function to get embeddings from model
    def encode(text):
        # TensorFlow does not require setting no_grad, as it does not have the same concept as PyTorch's no_grad
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()  # Extract embeddings and convert to numpy array

    # Asynchronously calculate embeddings
    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, encode, text) for text in data['contents']]
        embeddings = await asyncio.gather(*tasks)
    return pd.DataFrame([emb[0].tolist() for emb in embeddings], columns=[f'feature_{i}' for i in range(embeddings[0][0].shape[0])])


# Example Usage
async def prepare_data_from_csv(csv_filepath):
    train_data, test_data = await load_and_split_data(csv_filepath)
    train_data.to_csv('train/train_dataset.csv')
    test_data.to_csv('test/test_dataset.csv')
    return train_data, test_data

def load_into_faiss(data):
    embedding_columns = [col for col in data.columns if col.startswith('feature')]
    embeddings_matrix = data[embedding_columns].to_numpy(dtype='float32')
    d = embeddings_matrix.shape[1]

    # Create the FAISS index - using Flat index for simplicity
    index = faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap(index)

    ids = np.array(data.index.values)

    index.add_with_ids(embeddings_matrix, ids)

    return index

def extract_features_to_vector(df):
    # Filter columns that start with 'feature'
    feature_columns = [col for col in df.columns if col.startswith('feature')]
    
    # Extract the row and convert it to a NumPy array

    df['search_vector'] = df[feature_columns].apply(lambda row: row.tolist(), axis=1)
    
    return df

def find_similar_documents(search_vector, faiss_index, k=10):
    # Convert string to numpy array of the correct type and ensure it is 2D

    search_vector = np.array(search_vector, dtype='float32').reshape(1, -1)

    distances, indices = faiss_index.search(search_vector, k)

    train_df = pd.read_csv('train/train_dataset.csv')
    
    # Retrieve labels of the nearest neighbors
    nearest_labels = train_df.iloc[indices[0]]
    
    # Count occurrences of each class among the nearest neighbors
    class_count = Counter(nearest_labels['class'])
    
    return class_count

def classify_test_data(test_df, index):
    test_dict = test_df.to_dict(orient = 'records')
    for item in test_dict:
        class_counts = find_similar_documents(item['search_vector'], index)
        pred_class = class_counts.most_common(1)[0][0]
        item.update({'predicted_class': pred_class})
    return pd.DataFrame(test_dict)


if __name__ == '__main__':
    if 'unzipped_folders' not in os.listdir('data'):
        unzip_folder('data/trellis_assesment_ds.zip', 'data/unzipped_folders')
    if 'docs.csv' not in os.listdir('data'):
        write_texts_to_csv('data/unzipped_folders', 'data/docs.csv')
    if not ((os.path.exists('train')) or (os.path.exists('test'))):
        os.mkdir('train')
        os.mkdir('test')
        train_data, test_data = asyncio.run(prepare_data_from_csv('data/docs.csv'))
    if 'training_index.faiss' not in os.listdir('train'):
        train_data = pd.read_csv('train/train_dataset.csv')
        index = load_into_faiss(train_data)
        save_faiss_index(index, 'train/training_index.faiss')
        print('successfully saved index')
    else:
        index = faiss.read_index('train/training_index.faiss')
        print('successfully read index')

    test_data = pd.read_csv('test/test_dataset.csv')
    test_data = extract_features_to_vector(test_data)

    test_data = classify_test_data(test_data, index)

    y_true = test_data['class']
    y_pred = test_data['predicted_class']
    
    report = classification_report(y_true, y_pred)

    print(report)