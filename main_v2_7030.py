# @title Bibliotecas

import pandas as pd
import numpy as np
import flwr as fl
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, ndcg_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
from flwr_datasets.partitioner import PathologicalPartitioner, IidPartitioner, DirichletPartitioner, DistributionPartitioner
from datasets import Dataset

#import os
#os.environ["RAY_TMPDIR"] = ""

N_PARTITIONS = 300
N_CLASSES_PER_PARTITION = 12#200
DIRICHLET_ALPHA = 0.05
EXPERIMENT_CONFIG = 'NotShared' # Can be Shared, NotShared, Iid
SHARED_DATA_PERCENTAGE = 0.05
LOCAL_EPOCHS = 5
GLOBAL_EPOCHS = 3

def gen_distribution_array(N_PARTITIONS, N_UNIQUE_LABELS, N_UNIQUE_LABELS_PER_PARTITION, PREASSIGNED_NUM):
    distribution_array = np.zeros((N_UNIQUE_LABELS, N_PARTITIONS))
    j = 0
    for i in range(0, N_UNIQUE_LABELS, N_UNIQUE_LABELS_PER_PARTITION):
      distribution_array[i:i+N_UNIQUE_LABELS_PER_PARTITION, j] = 1
      if j >= N_PARTITIONS -1:
        j = 0
      else:
        j += 1
    return distribution_array

# @title Adequando métricas de precisão e recall para o formato do problema

#def precision(y_true, y_pred):
#    """
#    Nota >3 -> relevante
#    Nota <=3 -> n. relevante
#    """
#    y_true_rel = y_true > 3
#    y_pred_rel = y_pred > 3
#
#    return precision_score(y_true_rel, y_pred_rel, average='micro')
#
#def recall(y_true, y_pred):
#    """
#    Nota >3 -> relevante
#    Nota <=3 -> n. relevante
#    """
#    y_true_rel = y_true > 3
#    y_pred_rel = y_pred > 3
#
#    return recall_score(y_true_rel, y_pred_rel, average='micro')

def precision(y_true, y_pred, k=10):
    """
    precision@k
    """
    # Identificar indices das top-k
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    
    relevant = (y_true[top_k_idx] > 3)
    
    return np.sum(relevant) / k

def recall(y_true, y_pred, k=10):
    """
    recall@k
    """
    total_relevant = np.sum(y_true > 3)
    
    if total_relevant == 0:
        return 0.0  # evitar div 0
    
    # Identificar indices das top-k
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    
    relevant = (y_true[top_k_idx] > 3)
    
    # Recall@k = (número de relevantes nos top-k) / (total de relevantes)
    return np.sum(relevant) / total_relevant

def ndcg(y_true, y_pred):
    """
    Nota >3 -> relevante
    Nota <=3 -> n. relevante
    """
    y_true_rel = (y_true > 3).astype(int)

    return ndcg_score([y_true_rel], [np.asarray(y_pred).flatten()], k=10)

# {"mae": mae, "mse": loss, "rmse": rmse, "prec": prec, "rec": rec, "ndcg": ndcg_}
def weighted_average(metrics):
  maes = [num_examples * m["mae"] for num_examples, m in metrics]
  mses = [num_examples * m["mse"] for num_examples, m in metrics]
  rmses = [num_examples * m["rmse"] for num_examples, m in metrics]
  precs = [num_examples * m["prec"] for num_examples, m in metrics]
  recs = [num_examples * m["rec"] for num_examples, m in metrics]
  ndcgs = [num_examples * m["ndcg"] for num_examples, m in metrics]

  examples = [num_examples for num_examples, _ in metrics]
  # Aggregate and return custom metric (weighted average)
  return {"maes": sum(maes) / sum(examples),
          "mses": sum(mses) / sum(examples),
          "rmses": sum(rmses) / sum(examples),
          "precs": sum(precs) / sum(examples),
          "recs": sum(recs) / sum(examples),
          "ndcgs": sum(ndcgs) / sum(examples),}

# Flower Client Class
class MovieLensClient(fl.client.NumPyClient):
  def __init__(self, x_train, y_train, x_test, y_test):
    self.model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'root_mean_squared_error'])
    self.x_train, self.y_train = x_train, y_train
    self.x_test, self.y_test = x_test, y_test

  def get_parameters(self, config):
      return self.model.get_weights()

  def set_parameters(self, parameters):
      self.model.set_weights(parameters)

  def fit(self, parameters, config):
      self.model.set_weights(parameters)
      self.model.fit(self.x_train, self.y_train, epochs=LOCAL_EPOCHS, verbose=1)
      return self.model.get_weights(), len(self.x_train), {}

  def evaluate(self, parameters, config):
      self.model.set_weights(parameters)
      #loss, mae, rmse = self.model.evaluate(self.x_test, self.y_test, verbose=1)
      y_pred = self.model.predict(self.x_test)
      mae = mean_absolute_error(self.y_test, y_pred)
      rmse = root_mean_squared_error(self.y_test, y_pred)
      loss = rmse**2
      prec = precision(self.y_test, y_pred)
      rec  = recall(self.y_test, y_pred)
      ndcg_ = ndcg(self.y_test, y_pred)

      return loss, len(self.x_test), {"mae": mae, "mse": loss, "rmse": rmse, "prec": prec, "rec": rec, "ndcg": ndcg_}

# @title Leitura dos dados
print("Iniciando a leitura de dados...")

# Leitura ml-1m
# Python engine needed for sep regex
movies = pd.read_csv('ml-1m/movies.dat', sep="::", engine="python")
users = pd.read_csv('ml-1m/users.dat', sep="::", engine="python")
ratings = pd.read_csv('ml-1m/ratings.dat', sep="::", engine="python")

data = movies.merge(ratings, on="movieId").merge(users, on="userId")

#data['gender'] = data['gender'].map({'F': 0, 'M': 1})

# Categorical data conversion

# Genres
data['genres'] = data['genres'].str.split('|')
exploded_genres = data.explode('genres')
binary_genres = pd.get_dummies(exploded_genres['genres'], dtype=int)
binary_genres_grouped = binary_genres.groupby(exploded_genres.index).max()

# Occupation
#mapping = {
#    0:  "other or not specified",
#    1:  "academic/educator",
#    2:  "artist",
#    3:  "clerical/admin",
#    4:  "college/grad student",
#    5:  "customer service",
#    6:  "doctor/health care",
#    7:  "executive/managerial",
#    8:  "farmer",
#    9:  "homemaker",
#    10:  "K-12 student",
#    11:  "lawyer",
#    12:  "programmer",
#    13:  "retired",
#    14:  "sales/marketing",
#    15:  "scientist",
#    16:  "self-employed",
#    17:  "technician/engineer",
#    18:  "tradesman/craftsman",
#    19:  "unemployed",
#    20:  "writer",
#}
#binary_occupations = pd.get_dummies(data['occupation'], dtype=int).rename(columns=mapping)

# Normalization
data['timestamp'] = (data['timestamp'] - data['timestamp'].min()) / (data['timestamp'].max() - data['timestamp'].min())
data['userId'] = (data['userId'] - data['userId'].min()) / (data['userId'].max() - data['userId'].min())
data['movieId'] = (data['movieId'] - data['movieId'].min()) / (data['movieId'].max() - data['movieId'].min())
data['age'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())

# With occupation data
#df = pd.concat([data.drop(columns=['genres', 'occupation', 'zip-code', 'title']), binary_genres_grouped, binary_occupations], axis=1) # [1000209 rows x 45 columns]

# Without occupation data
df = pd.concat([data.drop(columns=['genres', 'occupation', 'zip-code', 'title', 'gender']), binary_genres_grouped], axis=1)

# Leitura ml-latest-small ou ml-32m
#links = pd.read_csv('ml-latest-small/links.csv')
#movies = pd.read_csv('ml-latest-small/movies.csv')
#ratings = pd.read_csv('ml-latest-small/ratings.csv')
#
#"""### Manipulação dos dados
#
#"""
#
#data = links.merge(movies, on="movieId").merge(ratings, on="movieId")
#data['genres'] = data['genres'].str.split('|')
#
## Normalização
#data['timestamp'] = (data['timestamp'] - data['timestamp'].min()) / (data['timestamp'].max() - data['timestamp'].min())
#data['userId'] = (data['userId'] - data['userId'].min()) / (data['userId'].max() - data['userId'].min())
#data['movieId'] = (data['movieId'] - data['movieId'].min()) / (data['movieId'].max() - data['movieId'].min())
#
#data = data.drop(columns=['imdbId', 'tmdbId', 'title'])
#
#generos_explodidos = data.explode('genres')
#
## Criar colunas binárias para cada gênero
#generos_binarios = pd.get_dummies(generos_explodidos['genres'], dtype=int)
#
## Reagrupar os dados ao nível do filme (somando as colunas binárias)
#generos_binarios_agrupados = generos_binarios.groupby(generos_explodidos.index).max()
#
## Concatenar as colunas binárias de volta ao DataFrame original
#df = pd.concat([data.drop(columns='genres'), generos_binarios_agrupados], axis=1)
#df = df.drop(columns='(no genres listed)')

print("Dados lidos com sucesso!")

print(df)

# apenas 1m de dados
#df = pd.concat([df, df, df, df, df, df, df, df, df, df], ignore_index=True)
#df = df.sample(n=1000000)

def main(i):
  train, test = train_test_split(df, test_size=0.3, random_state=900+i)

  # Partitioner for federated testing
  test_partitioner = IidPartitioner(num_partitions=N_PARTITIONS)
  test_fds = Dataset.from_pandas(test)
  test_partitioner.dataset = test_fds

  def load_test_partition(partition_id):
    partition_test = test_partitioner.load_partition(partition_id=partition_id)
    partition_np = partition_test.with_format("numpy")

    # Converter para numpy
    X = partition_np[list(partition_np.features.keys())[0]]
    for feature in list(partition_np.features.keys())[1:]:
      if feature != 'rating' and feature != '__index_level_0__':
        X = np.column_stack((X, partition_np[feature]))

    y = partition_np['rating']
    return X, y

  # Partitioner (partition_by=userId).
  #pathological_partitioner = PathologicalPartitioner(
  #    num_partitions=N_PARTITIONS, partition_by="userId", num_classes_per_partition=N_CLASSES_PER_PARTITION, class_assignment_mode="first-deterministic"
  #)

  #N_UNIQUE_LABELS = len(train['userId'].unique())
  #print("NUMBER OF UNIQUE LABELS: ", N_UNIQUE_LABELS)
  #N_UNIQUE_LABELS_PER_PARTITION = N_UNIQUE_LABELS // N_PARTITIONS
  #PREASSIGNED_NUM = 0
  #
  #distribution_array = gen_distribution_array(N_PARTITIONS, N_UNIQUE_LABELS, N_UNIQUE_LABELS_PER_PARTITION, PREASSIGNED_NUM)
  #
  #pathological_partitioner = DistributionPartitioner(
  #    distribution_array, 
  #    N_PARTITIONS,
  #    num_unique_labels_per_partition=N_UNIQUE_LABELS_PER_PARTITION,
  #    partition_by='userId',
  #    preassigned_num_samples_per_label=PREASSIGNED_NUM,
  #)

  pathological_partitioner = DirichletPartitioner(
      num_partitions=N_PARTITIONS, alpha=DIRICHLET_ALPHA, partition_by="userId"
  )

  fds = Dataset.from_pandas(train)
  pathological_partitioner.dataset = fds

  load_partition = None
  if EXPERIMENT_CONFIG=="Shared":

    # Creating shared subset
    x_shared = None
    y_shared = None
    for i in range(N_PARTITIONS):
      partition_np = pathological_partitioner.load_partition(partition_id=i).with_format("numpy")

      # Convert to numpy
      X = partition_np[list(partition_np.features.keys())[0]]
      for feature in list(partition_np.features.keys())[1:]:
        if feature != 'rating' and feature != '__index_level_0__':
          X = np.column_stack((X, partition_np[feature]))
      y = partition_np['rating']

      length = int(len(X)*SHARED_DATA_PERCENTAGE)
      if i==0:
        x_shared = X[:length]
        y_shared = y[:length]
      else:
        x_shared = np.concatenate((x_shared, X[:length]), axis=0)
        y_shared = np.concatenate((y_shared, y[:length]), axis=0)

    # Load a partition with shared data
    def f(partition_id):
      partition_pathological = pathological_partitioner.load_partition(partition_id=partition_id)
      partition_np = partition_pathological.with_format("numpy")

      # Convert to numpy
      X = partition_np[list(partition_np.features.keys())[0]]
      for feature in list(partition_np.features.keys())[1:]:
        if feature != 'rating' and feature != '__index_level_0__':
          X = np.column_stack((X, partition_np[feature]))

      y = partition_np['rating']
      length = int(len(X)*SHARED_DATA_PERCENTAGE)
      return np.concatenate((X[length:], x_shared), axis=0), np.concatenate((y[length:], y_shared), axis=0)
    
    load_partition = f

  elif EXPERIMENT_CONFIG=="NotShared":
  # No Shared data partition
    def f(partition_id):
      partition_pathological = pathological_partitioner.load_partition(partition_id=partition_id)
      partition_np = partition_pathological.with_format("numpy")

      # Convert to numpy
      X = partition_np[list(partition_np.features.keys())[0]]
      for feature in list(partition_np.features.keys())[1:]:
        if feature != 'rating' and feature != '__index_level_0__':
          X = np.column_stack((X, partition_np[feature]))

      y = partition_np['rating']
      return X, y

    load_partition = f

  elif EXPERIMENT_CONFIG=="Iid":
    # IID Partition (Reference Value)
    iid_partitioner = IidPartitioner(num_partitions=N_PARTITIONS)
    iid_fds = Dataset.from_pandas(train)
    iid_partitioner.dataset = iid_fds

    #Load an IID Training Partition
    def f(partition_id):
      partition_iid = iid_partitioner.load_partition(partition_id=partition_id)
      partition_np = partition_iid.with_format("numpy")

      # Convert to numpy
      X = partition_np[list(partition_np.features.keys())[0]]
      for feature in list(partition_np.features.keys())[1:]:
        if feature != 'rating' and feature != '__index_level_0__':
          X = np.column_stack((X, partition_np[feature]))

      y = partition_np['rating']
      return X, y

    load_partition = f

  else:
    raise Exception(f"EXPERIMENT_CONFIG={EXPERIMENT_CONFIG} is not valid!")

  # Client Generator
  def client_fn(cid):
    X, y = load_partition(int(cid))
    x_test, y_test = load_test_partition(int(cid))

    return MovieLensClient(X, y, x_test, y_test)

  strategy = fl.server.strategy.FedAvg(
      fraction_fit=1,  # 100% clients on train
      fraction_evaluate=1, # 100% clients on test
      min_fit_clients=N_PARTITIONS,
      min_evaluate_clients=N_PARTITIONS,
      min_available_clients=N_PARTITIONS,
      evaluate_metrics_aggregation_fn=weighted_average,
  )

  # Training data is stored in 'history'
  history = fl.simulation.start_simulation(
      client_fn=client_fn,
      num_clients=N_PARTITIONS,
      config=fl.server.ServerConfig(num_rounds=GLOBAL_EPOCHS),
      strategy=strategy,
      client_resources={"num_cpus": 1} # Uncomment to run with gpu. Be aware that a big number of clients may cause instability to your computer.
  )

  print(history)
  print(history.metrics_distributed)
  return history

N = 1 # Number of execs
SHARE_PERCENTAGE_PATHSUBSTRING = f"{SHARED_DATA_PERCENTAGE}percent-" if EXPERIMENT_CONFIG == "Shared" else ""
EXPERIMENT_KIND_PATHSUBSTRING = "WITH_shared_data" if EXPERIMENT_CONFIG == "Shared" else ("WITHOUT_shared_data" if EXPERIMENT_CONFIG == "NotShared" else "IID_REFERENCE")
REPORT_F_PATH = f"./results_n={N}-{N_PARTITIONS}_clients-{EXPERIMENT_KIND_PATHSUBSTRING}-{SHARE_PERCENTAGE_PATHSUBSTRING}fed-eval.csv"
with open(REPORT_F_PATH, "a+") as f:
  f.write("i,mae,mse,rmse,precision,recall,ndcg,time_in_seconds\n")

for i in range(N):
  # Execution
  print(f"Starting Exec {i+1}...")
  start = time.time()
  history = main(i)
  end = time.time()

  # Reports
  """
  {"maes": sum(maes) / sum(examples),
          "mses": sum(mses) / sum(examples),
          "rmses": sum(rmses) / sum(examples),
          "precs": sum(precs) / sum(examples),
          "recs": sum(recs) / sum(examples),
          "ndcgs": sum(ndcgs) / sum(examples),}
  
  """
  mae = history.metrics_distributed['maes'][-1][1]
  mse = history.metrics_distributed['mses'][-1][1]
  rmse = history.metrics_distributed['rmses'][-1][1]
  prec = history.metrics_distributed['precs'][-1][1]
  rec = history.metrics_distributed['recs'][-1][1]
  ndcg_ = history.metrics_distributed['ndcgs'][-1][1]
  with open(REPORT_F_PATH, "a+") as f:
    f.write(f"{i},{mae},{mse},{rmse},{prec},{rec},{ndcg_},{end-start}\n")