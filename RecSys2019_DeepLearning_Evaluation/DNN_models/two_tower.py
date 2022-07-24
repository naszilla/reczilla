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




class RetrievalModel(tfrs.models.Model):
  """MovieLens candidate generation model"""
 
  def __init__(self, query_model, candidate_model, retrieval_task_layer):
    super().__init__()
    self.query_model: tf.keras.Model = query_model
    self.candidate_model: tf.keras.Model = candidate_model
    self.retrieval_task_layer: tf.keras.layers.Layer = retrieval_task_layer
 
  def compute_loss(self, features, training=False) -> tf.Tensor:
    query_embeddings = self.query_model(features['user_id'])
    positive_candidate_embeddings = self.candidate_model(features["movie_id"])

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
    candidate_embeddings = self.candidate_model(features["movie_id"])
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
        self.process_attributes = False
        self.transform_data()
        self.get_models()

    def _compute_item_score(self, user_id_array, items_to_compute = None):

        afinity_scores, item_ids = self.brute_force_layer(tf.constant([user_id_array]))

        return afinity_scores

    def transform_data(self):
        self.ratings_df = pd.DataFrame.sparse.from_spmatrix(URM_train).unstack().reset_index()
        self.ratings_df.columns = ['item_id','user_id', 'rating']
        assert len(self.ratings_df) == self.URM_train.count_nonzero()
        self.ratings_dataset = tf.data.Dataset.from_tensor_slices(self.ratings_df.to_dict())
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
        
        self.training_dataset = self.ratings_dataset_shuffled.take(trainset_size)
        self.testing_dataset = self.ratings_dataset_shuffled.skip(trainset_size)

    def get_query_model(self):
        user_id_lookup_layer = StringLookup(mask_token=None)

        # StringLookup layer is a non-trainable layer and its state (the vocabulary)
        # must be constructed and set before training in a step called "adaptation".
        user_id_lookup_layer.adapt(
            training_dataset.map(
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
        if process_attributes:
            pass
            ## process attribute features and concatenate to embeddings
        
        # A model that takes raw string feature values (user_id) in and yields embeddings
        user_id_model = tf.keras.Sequential(
            [
                user_id_lookup_layer,
                user_id_embedding_layer
            ]
        )
        
        return user_id_model
    
    def get_candidate_model(self):
        item_id_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
        item_id_lookup_layer.adapt(ratings_trainset.map(lambda x: x['item_id']))

        # Same as user_id_embedding_dim to be able to measure the similarity
        item_id_embedding_dim = 32

        item_id_embedding_layer = tf.keras.layers.Embedding(
            input_dim=item_id_lookup_layer.vocab_size(),
            output_dim=item_id_embedding_dim
        )
 
        item_id_model = tf.keras.Sequential([
            item_id_lookup_layer,
            item_id_embedding_layer
            ]
        )
        return item_id_model

    def get_loss_layer(self):
        import tensorflow_recommenders as tfrs

        self.candidates_corpus_dataset = self.training_dataset.map(lambda item_id: item_id['item_id'])
        factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(
        # dataset of candidate embeddings from which candidates should be retrieved
        candidates=candidates_corpus_dataset.batch(128).map(
        candidate_model)
        )
        retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)
        return retrieval_task_layer

    def get_models(self):
        ### get individual models
        query_model = self.get_query_model()
        candidate_model = self.get_candidate_model()
        loss_layer = self.get_loss_layer()

        ### get retreival model
        retrieval_model = RetrievalModel(query_model,candidate_model,loss_layer)
        optimizer_step_size = 0.1
        retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=optimizer_step_size))
        self.retrieval_model = retrieval_model

        ### get ranking model
        ranking_model = RankingModel(query_model,candidate_model,loss_layer)
        optimizer_step_size = 0.1
        ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=optimizer_step_size))
        self.ranking_model = ranking_model
    
    def train_models(self):
        retrieval_cached_ratings_trainset = self.training_dataset.shuffle(100_000).batch(8192).cache()
        retrieval_cached_ratings_testset = self.testing_dataset.batch(4096).cache()
        
        num_epochs = 5 
        history_retreival = self.retrieval_model.fit(
            retrieval_cached_ratings_trainset,
            validation_data=retrieval_cached_ratings_testset,
            validation_freq=1,
            epochs=num_epochs
        )

        ranking_ratings_trainset = self.training_dataset.shuffle(100_000).batch(8192).cache()
        ranking_ratings_testset = self.testing_dataset.batch(4096).cache()


        history_ranking = self.ranking_model.fit(
            retrieval_cached_ratings_trainset,
            validation_data=retrieval_cached_ratings_testset,
            validation_freq=1,
            epochs=num_epochs
        )
        self.brute_force_layer = tfrs.layers.factorized_top_k.BruteForce(self.retrieval_model.query_model)

        self.brute_force_layer.index(
            self.candidates_corpus_dataset.batch(100).map(
                self.retrieval_model.candidate_model
            ),
            self.candidates_corpus_dataset
        )

    def save_model(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            retrieval_model_path = os.path.join(tmp_dir, "retrieval_model")


            scann_layer.save(
                retrieval_model_path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
                )
        