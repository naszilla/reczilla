from surprise import Dataset
from auto_surprise.engine import Engine
from data_handler import get_data

"""
Load the dataset. Choices:
'ml-100k', 'ml-1m', 'jester',
'book-crossing', 'dating', 'recipes'
"""
data = get_data('recipes')

# Intitialize auto surprise engine
engine = Engine(verbose=True)

# Start the trainer
best_algo, best_params, best_score, tasks = engine.train(
    data=data, 
    target_metric='test_rmse', 
    cpu_time_limit=60 * 60, 
    max_evals=100
)