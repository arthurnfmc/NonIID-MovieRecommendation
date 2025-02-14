import pandas as pd
import numpy as np
import flwr as fl
import time
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from flwr_datasets.partitioner import PathologicalPartitioner, IidPartitioner
from datasets import Dataset

N_PARTITIONS = 50
N_CLASSES_PER_PARTITION = 56
EXPERIMENT_CONFIG = 'Shared' # Can be Shared, NotShared, Iid
SHARED_DATA_PERCENTAGE = 0.05

"""

Reading data

"""
links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Data preview

#links.head()
#movies.head()
#ratings.head()
#data.head()

"""

Preprocess data

"""

data = links.merge(movies, on="movieId").merge(ratings, on="movieId")
data['genres'] = data['genres'].str.split('|')

# Normalização
data['timestamp'] = (data['timestamp'] - data['timestamp'].min()) / (data['timestamp'].max() - data['timestamp'].min())
data['userId'] = (data['userId'] - data['userId'].min()) / (data['userId'].max() - data['userId'].min())
data['movieId'] = (data['movieId'] - data['movieId'].min()) / (data['movieId'].max() - data['movieId'].min())

data = data.drop(columns=['imdbId', 'tmdbId', 'title'])

exploded_genres = data.explode('genres')

# Create binary columns for each genre
binary_genres = pd.get_dummies(exploded_genres['genres'], dtype=int)
binary_genres_grouped = binary_genres.groupby(exploded_genres.index).max()

# Concat binary columns back to original dataframe
df = pd.concat([data.drop(columns='genres'), binary_genres_grouped], axis=1)
df = df.drop(columns='(no genres listed)')

# Dataframe visualization
#print(df)

"""

Actual experiment code

"""

def main():

  train, test = train_test_split(df, test_size=0.3)

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
  pathological_partitioner = PathologicalPartitioner(
      num_partitions=N_PARTITIONS, partition_by="userId", num_classes_per_partition=N_CLASSES_PER_PARTITION, class_assignment_mode="first-deterministic"
  )

  # Loading training data
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

  #def weighted_average(metrics):
  #    maes = [num_examples * m["mae"] for num_examples, m in metrics]
  #    mses = [num_examples * m["mse"] for num_examples, m in metrics]
  #    examples = [num_examples for num_examples, _ in metrics]
  #
  #    # Aggregate and return custom metric (weighted average)
  #    return {"maes": sum(maes) / sum(examples),
  #            "mses": sum(mses) / sum(examples)}
  
  def average(metrics):
    maes = [m["mae"] for _, m in metrics]
    mses = [m["mse"] for _, m in metrics]

    return {
        "maes": sum(maes) / len(maes),
        "mses": sum(mses) / len(mses),
    }

  # Flower Client Class
  class MovieLensClient(fl.client.NumPyClient):
      def __init__(self, x_train, y_train, x_test, y_test):
          self.model = keras.Sequential([
              layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
              layers.Dense(32, activation='relu'),
              layers.Dense(1)
          ])
          self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
          self.x_train, self.y_train = x_train, y_train
          self.x_test, self.y_test = x_test, y_test

      def get_parameters(self, config):
          return self.model.get_weights()

      def set_parameters(self, parameters):
          self.model.set_weights(parameters)

      def fit(self, parameters, config):
          self.model.set_weights(parameters)
          self.model.fit(self.x_train, self.y_train, epochs=5, verbose=1)
          return self.model.get_weights(), len(self.x_train), {}

      def evaluate(self, parameters, config):
          self.model.set_weights(parameters)
          loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=1)
          return loss, len(self.x_test), {"mae": mae, "mse": loss}

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
      evaluate_metrics_aggregation_fn=average,
  )

  # Training data is stored in 'history'
  history = fl.simulation.start_simulation(
      client_fn=client_fn,
      num_clients=N_PARTITIONS,
      config=fl.server.ServerConfig(num_rounds=3),
      strategy=strategy,
      #client_resources={"num_cpus": 1, "num_gpus": 1} # Uncomment to run with gpu. Be aware that a big number of clients may cause instability to your computer.
  )

  print(history)
  print(history.metrics_distributed)
  return history

N = 33 # Number of executions (33 magic number)
SHARE_PERCENTAGE_PATHSUBSTRING = f"{SHARED_DATA_PERCENTAGE}percent-" if EXPERIMENT_CONFIG == "Shared" else ""
EXPERIMENT_KIND_PATHSUBSTRING = "WITH_shared_data" if EXPERIMENT_CONFIG == "Shared" else ("WITHOUT_shared_data" if EXPERIMENT_CONFIG == "NotShared" else "IID_REFERENCE")
REPORT_F_PATH = f"./results_n={N}-{N_PARTITIONS}_clients-{EXPERIMENT_KIND_PATHSUBSTRING}-{SHARE_PERCENTAGE_PATHSUBSTRING}fed-eval.csv"
with open(REPORT_F_PATH, "a+") as f:
  f.write("i,mae,mse,time_in_seconds\n")

for i in range(N):
  # Execution
  start = time.time()
  history = main()
  end = time.time()

  # Reports
  mae = history.metrics_distributed['maes'][-1][1]
  mse = history.metrics_distributed['mses'][-1][1]
  with open(REPORT_F_PATH, "a+") as f:
    f.write(f"{i},{mae},{mse},{end-start}\n")