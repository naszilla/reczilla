from typing import Dict, List, Tuple, Optional
import logging
import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Activation, Concatenate, Dense, Embedding, Add, GlobalMaxPooling1D, GlobalAveragePooling1D, Dot, StringLookup
)

from types import SimpleNamespace
from Data_manager.DataSplitter_global_timestamp import DataSplitter_global_timestamp

from Base.BaseRecommender import BaseRecommender
from Base.BaseTempFolder import BaseTempFolder
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.DataIO import DataIO

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




class RetrievalModel(tfrs.models.Model):
  """MovieLens candidate generation model"""
 
  def __init__(self, query_model, candidate_model, retrieval_task_layer):
    super().__init__()
    self.query_model: tf.keras.Model = query_model
    self.candidate_model: tf.keras.Model = candidate_model
    self.retrieval_task_layer: tf.keras.layers.Layer = retrieval_task_layer
 
  def compute_loss(self, features, training=False) -> tf.Tensor:
    query_embeddings = self.query_model(features['user_id'])
    positive_candidate_embeddings = self.candidate_model(features["item_id"])

    loss = self.retrieval_task_layer(
        query_embeddings,
        positive_candidate_embeddings
        # ,compute_metrics=not training  # To speed up training
    )
    return loss


class RankingModel(tfrs.models.Model):
  """MovieLens ranking model"""

  def __init__(self, query_model, candidate_model):
    super().__init__()

    self.query_model: tf.keras.Model = query_model
    self.candidate_model: tf.keras.Model = candidate_model
    self.rating_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ]
    )
    self.ranking_task_layer: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )


  def compute_loss(self, features, training=False) -> tf.Tensor:
    query_embeddings = self.query_model(features['user_id'])
    candidate_embeddings = self.candidate_model(features["item_id"])
    rating_predictions = self.rating_model(
        tf.concat(
            [query_embeddings, candidate_embeddings],
            axis=1
        )
        # We could use `tf.keras.layers.Concatenate(axis=1)([x, y])`
    )

    loss = self.ranking_task_layer(
        predictions=rating_predictions,
        labels=features["user_rating"]
    )
    return loss







class Two_Tower_Recommender(BaseRecommender):

    RECOMMENDER_NAME = "Two_Tower_Recommender"

    def __init__(self, URM_train):
        super(Two_Tower_Recommender, self).__init__(URM_train)
        self.task = 'Ranking'
        self.process_attributes = False
        self.transform_data()
        self.get_models()

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        import numpy as np
        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = np.dot(self.user_embeddings[user_id_array,:], self.item_embeddings[items_to_compute,:].T)
        else:
            item_scores = np.dot(self.user_embeddings[user_id_array,:], self.item_embeddings.T)

        return item_scores

    def transform_data(self):
        import pandas as pd
        self.ratings_df = pd.DataFrame({'item_id':self.URM_train.tocoo().col,
                                        'user_id':self.URM_train.tocoo().row,
                                        'user_rating':self.URM_train.tocoo().data})
        self.ratings_df['item_id'] = self.ratings_df['item_id'].astype(str) 
        self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(str) 
        self.ratings_df = self.ratings_df[self.ratings_df.user_rating!=0]
        assert len(self.ratings_df) == self.URM_train.count_nonzero()
        self.ratings_dataset = tf.data.Dataset.from_tensor_slices(self.ratings_df.to_dict(orient="list"))
        self.ratings_dataset = self.ratings_dataset.map(lambda rating: {
                                                    # `user_id` is useful as a user identifier.
                                                    'user_id': rating['user_id'],
                                                    # `movie_id` is useful as a movie identifier.
                                                    'item_id': rating['item_id'],
                                                    # `movie_title` is useful as a textual information about the movie.
                                                    # `user_rating` shows the user's level of interest to a movie.
                                                    'user_rating': rating['user_rating'],
                                                    # `timestamp` will allow us to model the effect of time.
            }
        )
        self.trainset_size = 0.8 * self.ratings_dataset.__len__().numpy()
        self.ratings_dataset_shuffled = self.ratings_dataset.shuffle(
            # the new dataset will be sampled from a buffer window of first `buffer_size`
            # elements of the dataset
            buffer_size=100_000,
            # set the random seed that will be used to create the distribution.
            seed=42,
            # `list(dataset.as_numpy_iterator()` yields different result for each call
            # Because reshuffle_each_iteration defaults to True.
            reshuffle_each_iteration=False
        )
        
        self.training_dataset = self.ratings_dataset_shuffled.take(self.trainset_size)
        self.testing_dataset = self.ratings_dataset_shuffled.skip(self.trainset_size)
        print("dataset set")

    def get_query_model(self):
        user_id_lookup_layer = StringLookup(mask_token=None)

        # StringLookup layer is a non-trainable layer and its state (the vocabulary)
        # must be constructed and set before training in a step called "adaptation".
        user_id_lookup_layer.adapt(
            self.training_dataset.map(
                lambda x: x['user_id']
            )
        )

        user_id_embedding_dim = 32
        # The larger it is, the higher the capacity of the model, but the slower it is
        # to fit and serve and more prone to overfitting.

        user_id_embedding_layer = tf.keras.layers.Embedding(
            # Size of the vocabulary
            input_dim=user_id_lookup_layer.vocab_size(),
            # Dimension of the dense embedding
            output_dim=user_id_embedding_dim
        )
        if self.process_attributes:
            pass
            ## process attribute features and concatenate to embeddings
        
        # A model that takes raw string feature values (user_id) in and yields embeddings
        self.query_model = tf.keras.Sequential(
            [
                user_id_lookup_layer,
                user_id_embedding_layer
            ]
        )
        
    
    def get_candidate_model(self):
        item_id_lookup_layer = StringLookup(mask_token=None)
        item_id_lookup_layer.adapt(self.training_dataset.map(lambda x: x['item_id']))

        # Same as user_id_embedding_dim to be able to measure the similarity
        item_id_embedding_dim = 32

        item_id_embedding_layer = tf.keras.layers.Embedding(
            input_dim=item_id_lookup_layer.vocab_size(),
            output_dim=item_id_embedding_dim
        )
 
        self.candidate_model = tf.keras.Sequential([
            item_id_lookup_layer,
            item_id_embedding_layer
            ]
        )

    def get_loss_layer(self):
        import tensorflow_recommenders as tfrs

        self.candidates_corpus_dataset = self.training_dataset.map(lambda item_id: item_id['item_id'])
        factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(
        # dataset of candidate embeddings from which candidates should be retrieved
        candidates=self.candidates_corpus_dataset.batch(128).map(
        self.candidate_model)
        )
        self.loss_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)

    def get_models(self):
        ### get individual models
        self.get_query_model()
        self.get_candidate_model()
        self.get_loss_layer()

        ### get retreival model
        retrieval_model = RetrievalModel(self.query_model,self.candidate_model,self.loss_layer)
        optimizer_step_size = 0.1
        retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=optimizer_step_size))
        self.retrieval_model = retrieval_model

        ### get ranking model
        ranking_model = RankingModel(self.query_model,self.candidate_model)
        optimizer_step_size = 0.1
        ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=optimizer_step_size))
        self.ranking_model = ranking_model
    
    def fit(self):
        num_epochs = 5
        if self.task=='Retreival':
            retrieval_cached_ratings_trainset = self.training_dataset.shuffle(100_000).batch(32).cache()
            retrieval_cached_ratings_testset = self.testing_dataset.batch(4096).cache()
            
            history_retreival = self.retrieval_model.fit(
                retrieval_cached_ratings_trainset,
                validation_data=retrieval_cached_ratings_testset,
                validation_freq=1,
                epochs=num_epochs
            )

        elif self.task=='Ranking':
            retrieval_cached_ratings_trainset = self.training_dataset.shuffle(100_000).batch(32).cache()
            retrieval_cached_ratings_testset = self.testing_dataset.batch(4096).cache()
            history_ranking = self.ranking_model.fit(
                retrieval_cached_ratings_trainset,
                validation_data=retrieval_cached_ratings_testset,
                validation_freq=1,
                epochs=num_epochs
            )
        num_users, num_items = self.URM_train.shape
        self.user_embeddings = self.query_model(tf.constant(np.arange(num_users).astype(str))).numpy()
        self.item_embeddings = self.candidate_model(tf.constant(np.arange(num_items).astype(str))).numpy()
            

    def save_model(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            query_model_path = os.path.join(tmp_dir, "query_model")
            self.query_model.save(
                query_model_path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["query"])
                )
            candidate_model_path = os.path.join(tmp_dir, "candidate_model")
            self.query_model.save(
                query_model_path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["candidate"])
                )

    def load_model(self):
        query_model_path = os.path.join(tmp_dir, "query_model")
        candidate_model_path = os.path.join(tmp_dir, "candidate_model")
        self.query_model = tf.keras.models.load_model(query_model_path)
        self.candidate_model = tf.keras.models.load_model(candidate_model_path)