import functools
import tensorflow as tf


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
    def __init__(self, input_user, input_item, output, num_users, num_items, layers, MLP=False):
        self.input_user = input_user
        self.input_item = input_item
        self.output = output
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = layers[-1]/2
        self.layers = layers
        self.MLP = MLP
        self.predict
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        with tf.name_scope("embedding"):
            embedding_users = tf.get_variable("embedding_users", [self.num_users, self.embedding_size])
            embedding_items = tf.get_variable("embedding_items", [self.num_items, self.embedding_size])
            self.embedding_users = embedding_users
            self.embedding_items = embedding_items
            user_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1)
            item_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1)
            # scale_embedding_users = tf.reduce_sum(embedding_users*embedding_users, axis=1)/self.embedding_size
            # scale_embedding_items = tf.reduce_sum(embedding_items * embedding_items, axis=1) / self.embedding_size

            # merge_embedding = tf.concat([user_embedding, item_embedding], axis=1, name="merge_embedding")
            # tf.summary.histogram("embedding_users",embedding_users)
            # tf.summary.histogram("embedding_items",embedding_items)

            # embedding_users_g = tf.get_variable("embedding_users_GMF", [self.num_users, self.embedding_size])
            # embedding_items_g = tf.get_variable("embedding_items_GMF", [self.num_items, self.embedding_size])
            # self.embedding_items_g = embedding_items_g
            # user_embedding_g = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users_g, self.input_user), axis=1)
            # item_embedding_g = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items_g, self.input_item), axis=1)
            # GMF_embed = user_embedding_g*item_embedding_g

            inter1 = tf.reduce_sum(user_embedding * item_embedding, axis=1, keep_dims=True)
            # inter2 = tf.reduce_sum(user_embedding * item_embedding_g, axis=1, keep_dims=True)
            # inter = tf.concat([inter1, inter2], axis=1)
            # inter = tf.reduce_max(inter, axis=1, keep_dims=True)
            # tf.summary.histogram("user_embedding", user_embedding)
        # x = merge_embedding
        # with tf.name_scope("fc"):
        #     for i in xrange(len(self.layers) - 1):
        #         x = tf.contrib.slim.fully_connected(x, self.layers[i + 1])
        #     if not self.MLP:
        #         x = tf.concat([x, GMF_embed], axis=1, name = "concat_embedding")
        #         print ("MLP is False, Running NeuMF")
        #     x = tf.contrib.slim.fully_connected(x, 1, tf.identity)
        return inter1,item_embedding

    @define_scope
    def optimize(self):
        with tf.name_scope("optimize"):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict[0],
                                                                 labels=self.output, name="cross_entropy")
            # tf.summary.scalar("cross_entropy", loss)
            # sim_loss = tf.reduce_mean(self.predict[1] * self.predict[2], axis=1) / (
            # tf.norm(self.predict[1], axis=1) * tf.norm(self.predict[2], axis=1))
            # loss = tf.reduce_sum(cross_loss, 1) + sim_loss
            # optimizer = tf.train.AdamOptimizer()
            optimizer = tf.train.GradientDescentOptimizer(0.01)
        return optimizer.minimize(loss), tf.reduce_mean(loss),tf.Variable(0),tf.Variable(0)

    @define_scope
    def status(self):
        embedding_users = self.embedding_users
        embedding_items = self.embedding_items
        scale_embedding_users = tf.reduce_sum(embedding_users * embedding_users, axis=1) / self.embedding_size
        scale_embedding_items = tf.reduce_sum(embedding_items * embedding_items, axis=1) / self.embedding_size
        return scale_embedding_users, scale_embedding_items
