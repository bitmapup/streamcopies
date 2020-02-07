import os
import logging
import tqdm
import math
import random
import time
import joblib
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import utils, metrics
from imblearn.tensorflow import balanced_batch_generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#path = '/data/irene.unceta/results/graphs/'
path = '/Users/irene/Documents/Projects/graphs_bis/results/'

class Original(object):

    def __init__(self,
             hidden_layers,
             n_classes,
             loss_opt=tf.train.AdamOptimizer,
             class_opt=tf.train.AdamOptimizer,
             neuron='relu',
             class_weight=None):
    
        assert len(hidden_layers) > 1

        self.logger = logging.getLogger(__name__)
 
        self._total_epoch = 0
        self._n_classes = n_classes

        self._create_graph(hidden_layers, loss_opt, class_opt,
                           n_classes, class_weight, neuron)

        self._saver = tf.train.Saver()

        self._sess = tf.InteractiveSession()
        self._sess.run(self._init)
    
    def _create_graph(self, hidden_layers, loss_opt, class_opt,
                      n_classes, class_weight, neuron):

        tf.reset_default_graph()
        
        with tf.variable_scope('placeholder'):
            X_ = tf.placeholder('float', [None, hidden_layers[0]], name='X_')
            y_ = tf.placeholder('float', name='y_')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            is_train = tf.placeholder(tf.bool, name="is_train")
        
        # Batch normalization
        if neuron == 'relu':
            X_norm = tf.layers.batch_normalization(X_, training=is_train)
        else:
            X_norm = X_
        
        # Network architecture
        c = 0.1
        with tf.variable_scope('network'):
            tmp = tf.nn.dropout(X_norm, keep_prob)
            for idx_1, (from_, to_) in enumerate(zip(hidden_layers, hidden_layers[1:])):
                with tf.variable_scope('layer_{}'.format(idx_1)):
                    b = tf.get_variable('bias', initializer=c*tf.random_normal([to_]))
                    w = tf.get_variable('weights', initializer=c*tf.random_normal([from_, to_]))
                    if neuron == 'relu':
                        tmp = tf.nn.relu(tf.add(tf.matmul(tmp, w), b))
                    else:
                        tmp = tf.nn.selu(tf.add(tf.matmul(tmp, w), b))
                    #tmp = tf.nn.dropout(tmp, keep_prob)
            
                    output = tf.identity(tmp, name='output')
            
            # Classification layer
            with tf.variable_scope('out'):
                b = tf.get_variable('bias', initializer=c*tf.random_normal([n_classes]))
                w = tf.get_variable('weights', initializer=c*tf.random_normal([hidden_layers[-1], n_classes]))
                raw_pred = tf.add(tf.matmul(output, w), b, name='raw_predictions')

            #raw_pred = tf.identity(output, name='raw_predictions')
        
        # Loss
        with tf.variable_scope('loss'):
            class_cost = tf.losses.softmax_cross_entropy(logits=raw_pred, onehot_labels=y_)
            mean_class_cost = tf.reduce_mean(class_cost, name='mean_class_cost')
        
        # Train
        with tf.variable_scope('train'):
            loss_opt(learning_rate).minimize(mean_class_cost, name='minimize')
        
        # Accuracy metrics
        with tf.variable_scope('metrics'):
            predictions = tf.argmax(raw_pred, axis=1, name='predictions')
            
        self._init = tf.global_variables_initializer()
        
    def _prepare_y(self, y):
        
        if y is not None:
            assert self._n_classes > 1, 'At least 2 classes are needed to perform classification'
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.shape[1] == 1:
                assert self._n_classes == np.unique(y).size, \
                        'The number of unique labels ({}) should match the given n_classes ({})'.format(
                                np.unique(y).size, self._n_classes)
                assert int(np.unique(y).min()) == 0, \
                        'Min label ({}) should be 0'.format(int(np.unique(y).min()))
                assert int(np.unique(y).max()) == self._n_classes -1, \
                        'Max label ({}) should match the given n_classes -1 ({})'.format(
                                int(np.unique(y).max()), self._n_classes -1)
                _y = np.zeros((y.shape[0], self._n_classes))
                _y[np.arange(y.shape[0]), y[:,0].astype('int')] = 1
                y = _y

        return y
                
    def fit(self, X, y,
            epochs=10, batch_size=64, learning_rate=1e-3,
            d_lr=False, lr_dc=None, lr_dc_rate=0.9, keep_prob=1, 
            total_batch=None, save_dict=None):
        
        y = self._prepare_y(y)

        self._loss = []
        
        bar = tqdm.tqdm(iterable=range(1, epochs+1), leave=True)
        for epoch in bar:
            
            self._total_epoch += 1
            
            idx = list(range(X.shape[0]))
            random.shuffle(idx)

            training_generator, total_batch = balanced_batch_generator(X, y, sample_weight=None, sampler=None,
                                                                       batch_size=batch_size, random_state=None)

            epoch_loss = 0.
            
            pbar =  tqdm.tqdm(iterable=range(total_batch), leave=True, miniters=1, desc= '{} epoch'.format(epoch))
            for t in pbar:

                batch_x, batch_y = next(training_generator)

                _, lc = self._sess.run([tf.get_default_graph().get_operation_by_name('train/minimize'),
                                        tf.get_default_graph().get_tensor_by_name('loss/mean_class_cost:0')],
                                        feed_dict={'placeholder/X_:0': batch_x,
                                                   'placeholder/y_:0': batch_y,
                                                   'placeholder/keep_prob:0': keep_prob,
                                                   'placeholder/learning_rate:0': learning_rate,
                                                   'placeholder/is_train:0': True})
                
                self._loss.append(lc)
                epoch_loss += lc
                info={'loss': epoch_loss/(t+1)}
                pbar.set_postfix(info)
            
            bar.write('')

        self.save_graph(save_dict)
    
    def save_graph(self, path):
        fname = 'model.ckpt'
        save_path = self._saver.save(self._sess, path+fname)
        
    def predict(self, X):
                
        return self._sess.run(tf.get_default_graph().get_tensor_by_name('metrics/predictions:0'),
                                feed_dict={'placeholder/X_:0': X,
                                           'placeholder/keep_prob:0': 1,
                                           'placeholder/is_train:0': False})

    def predict_raw(self, X):
        
        return self._sess.run(tf.get_default_graph().get_tensor_by_name('network/out/raw_predictions:0'),
                              feed_dict={'placeholder/X_:0': X,
                              'placeholder/keep_prob:0': 1,
                              'placeholder/is_train:0': False})

    def predict_layer(self, X, layer):
        
        return self._sess.run(tf.get_default_graph().get_tensor_by_name('network/layer_{}/output:0'.format(layer)),
                              feed_dict={'placeholder/X_:0': X,
                              'placeholder/keep_prob:0': 1,
                              'placeholder/is_train:0': False})

    def retore(self, path):
        
        saver = tf.train.import_meta_graph(path+'model.ckpt.meta')
        saver.restore(self._sess, path+'model.ckpt')

if __name__ == '__main__':

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    fh = logging.FileHandler(path+'models/non_transactional/net_logs.log')
    fh.setLevel(logging.DEBUG)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter and add it to both handlers
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.info('Loading data...')
    
    # Load data
    data = joblib.load(path+'NT_data.pkl')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info('DONE!')
    
    epochs = 10
    batch_size = 32
    keep_prob = 0.9
    neuron = 'selu'
    hidden_layers = [X_train.shape[1], 32, 16]

    logger.info('epochs: {}'.format(epochs))
    logger.info('batch_size: {}'.format(batch_size))
    logger.info('neuron: {}'.format(neuron))
    logger.info('keep_prob: {}'.format(keep_prob))
    logger.info('hidden_layers: {}'.format(hidden_layers))

    model = Original(hidden_layers=hidden_layers,
                n_classes=2, neuron=neuron)
    model.fit(X_train, y_train, batch_size=batch_size, keep_prob=keep_prob, epochs=epochs, total_batch=None, save_dict=path+'models/non_transactional/net_')

    y_pred = model.predict(X_test)
    joblib.dump(y_pred, path+'models/non_transactional/net_y_pred_test.pkl' )
    test_acc = metrics.balanced_accuracy_score(y_test, y_pred)
    logger.info('test acc: {}'.format(test_acc))
    logger.info('{}'.format(metrics.confusion_matrix(y_test, y_pred)))
    logger.info('{}'.format(metrics.classification_report(y_test, y_pred)))

    #X_pred_test_raw = model.predict_raw(X_test)
    #X_pred_train_raw = model.predict_raw(X_train)
    #joblib.dump({'X_train': X_pred_train_raw,
    #             'y_train': y_train,
    #             'X_test': X_pred_test_raw,
    #             'y_test': y_test},
    #             path+'models/non_transactional/net_data_raw.pkl')

    #X_pred_test_layer16 = model.predict_layer(X_test, len(hidden_layers)-2)
    #X_pred_train_layer16 = model.predict_layer(X_train, len(hidden_layers)-2)
    #joblib.dump({'X_train': X_pred_train_layer16,
    #             'y_train': y_train,
    #             'X_test': X_pred_test_layer16,
    #             'y_test': y_test},
    #             path+'models/non_transactional/net_data_layer16.pkl')
    
