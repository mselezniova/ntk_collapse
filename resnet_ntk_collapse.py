import sys, getopt
from functools import partial
from typing import Any, Callable, Sequence, Tuple
import itertools

import jax
from jax import lax, random, numpy as jnp

import flax
from flax import linen as nn
from flax.training import common_utils
from flax.training import train_state  
import optax

import pickle

import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm

import gc

tf.config.experimental.set_visible_devices([], "GPU")

# -----------------------------------
# -----------------------------------
# HYPERPARAMETERS

opts, args = getopt.getopt(sys.argv[1:],"p:f:l:d:i:",["path=","filename=",'lr=','dataset=','init='])
for opt, arg in opts:
    if opt=="--dataset":
        DATASET = arg
    elif opt=="--lr":
        LEARNING_RATE = float(arg)
    elif opt=="--init":
        KERNEL_INIT = arg # 'flax', 'torch' or 'normal fan_out'
    elif opt=='--path':
        PATH = arg
    elif opt=='--filename':
        FILE = arg

print('Architecture: ResNet20')
print('Dataset:', DATASET)
print('Initialization:', KERNEL_INIT)
print('Initial learning rate:', LEARNING_RATE)
print('Output file:', PATH+FILE)

NUM_CLASSES = 10 # number of classes
BIAS = True # use bias in the classifier layer

WEIGHTS_DECAY = 0.0005
LOSS = 'MSE' # 'CE' for cross-entropy or 'MSE' for squared loss

num_epochs = 400 
batch_size = 120

m = 12 # number of sampled elements from each class to sample kernel values
EPOCH_STEP = 1 # compute metrics every EPOCH_STEP epochs

# -----------------------------------
# -----------------------------------
# LOADING DATA

train_ds = tfds.load(DATASET, split='train')
test_ds = tfds.load(DATASET, split='test')

train_ds = train_ds.map(lambda sample: {'image': tf.cast(sample['image'],
                                                       tf.float32) / 255.,
                                      'label': sample['label']}) # normalize train set
test_ds = test_ds.map(lambda sample: {'image': tf.cast(sample['image'],
                                                     tf.float32) / 255.,
                                    'label': sample['label']}) # normalize test set

test_ds = test_ds.shuffle(1024, seed=0).batch(batch_size, drop_remainder=True).cache().prefetch(10)

min_samples_per_class = {'mnist': 5412,        # number of samples in each class
                         'FashionMNIST': 6000,
                         'cifar10': 5000
                        }
image_size = {'mnist': (1,28,28,1),       
              'FashionMNIST': (1,28,28,1),
              'cifar10': (1,32,32,3)
                        }

DATASET_TOTAL_SAMPLES_PER_CLASS = min_samples_per_class[DATASET] 
DATASET_IMAGE_SIZE = image_size[DATASET]

NUM_H = 64 # number of features

# -----------------------------------
# -----------------------------------
# Initializers

if KERNEL_INIT == 'normal_fan_out':
    kernel_init_1 = kernel_init_2 = nn.initializers.variance_scaling(scale = 2.0, mode='fan_out', distribution='normal')
    bias_init = nn.initializers.zeros_init()
elif KERNEL_INIT == 'flax':
    kernel_init_1 = kernel_init_2 = nn.initializers.lecun_normal()
    bias_init = nn.initializers.zeros_init()
elif KERNEL_INIT == 'torch':
    kernel_init_1 = nn.initializers.variance_scaling(scale = 1.0, mode='fan_in', distribution='uniform')
    kernel_init_2 = nn.initializers.variance_scaling(scale = 1./9., mode='fan_in', distribution='uniform')
    bias_init = nn.initializers.uniform(scale=jnp.sqrt(1./NUM_H))
    

# -----------------------------------
# -----------------------------------
# Architecture

class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False 

    @nn.compact
    def __call__(self, x, train=True):
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=kernel_init_2,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=kernel_init_2,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=kernel_init_1)(x)

        x_out = self.act_fn(z + x)
        return x_out


class ResNetFeatures(nn.Module):
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=kernel_init_2, use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)

        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0)
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        x = x.mean(axis=(1, 2))
        return x
    
class ResNetClassifier(nn.Module):
    num_classes: int    
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_classes, use_bias=BIAS,
                     kernel_init=kernel_init_1, 
                     bias_init=bias_init)(x)
        x = jnp.asarray(x) 
        return x

    
class ResNet(nn.Module):
    num_classes: int
    act_fn : callable = nn.relu
    block_class : nn.Module = ResNetBlock
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)
        
    def setup(self):
        self.features = ResNetFeatures(act_fn=self.act_fn, 
                                       block_class=self.block_class,
                                       num_blocks=self.num_blocks,
                                       c_hidden=self.c_hidden)
        self.classifier = ResNetClassifier(num_classes=self.num_classes)

    def __call__(self, x, train: bool = True):
        x = self.features(x, train)
        x = self.classifier(x)
        return x

# -----------------------------------
# -----------------------------------
# Training tools


class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(rng, model, image_size = DATASET_IMAGE_SIZE, 
                       lr=optax.exponential_decay(init_value=0.05, 
                                                  transition_steps=40, decay_rate=0.1)):

    variables = model.init({'params': rng}, jnp.ones(image_size))
    params, batch_stats = variables['params'], variables['batch_stats']
    
    tx = optax.sgd(
      learning_rate=lr,
      momentum=0.9,
      nesterov=True
    )
    
    state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats)
    
    return state


@jax.jit
def mse_loss(predictions, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    squared_errors = optax.l2_loss(predictions=predictions, targets=one_hot_labels)
    return jnp.mean(squared_errors)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            mutable=['batch_stats'])

        if LOSS == 'CE':
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=predictions, labels=batch['label']).mean()
        elif LOSS == 'MSE':
            loss = mse_loss(predictions, batch['label'])

        
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        
        weight_l2 = sum(jnp.sum(x ** 2)
                         for x in weight_penalty_params
                         if x.ndim > 1)
        weight_penalty = WEIGHTS_DECAY * 0.5 * weight_l2

        loss = loss + weight_penalty
        return loss, (new_model_state, predictions)

    step = state.step

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    
    new_model_state, predictions = aux[1]

    new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats'])

    return new_state, predictions


# -----------------------------------
# -----------------------------------
# Metrics

@jax.jit
def features(params, batch_stats, x):
    return model.apply({"params": params, 'batch_stats': batch_stats}, x,
                                 method=lambda module, x: module.features(x, train=False))

def weights(params):
    return params['classifier']['Dense_0']['kernel'].T

def biases(params):
    return params['classifier']['Dense_0']['bias']

@jax.jit
def H1(state, x_by_class):
    
    def h_class_mean(x):
        return jnp.mean(model.apply({"params": state.params, 'batch_stats': state.batch_stats}, x,
                            method=lambda module, x: module.features(x, train=False)), axis=0)
    
    return jax.vmap(h_class_mean)(x_by_class).T


def Q2(size_p_class = 100):
    sq_m = jnp.sqrt(size_p_class)
    B_norm = sq_m*(sq_m+1)
    B_dir = jnp.array([[sq_m+1]+i*[1.]+[-sq_m*(sq_m+1)+1]+(size_p_class-i-2)*[1.] 
                       for i in range(size_p_class-1)]).T
    
    return jnp.kron(jnp.eye(NUM_CLASSES), B_dir), B_norm


def H2(state, x_by_class):
    X = jnp.vstack(x_by_class)
    m = x_by_class[0].shape[0]
    
    H = model.apply({"params": state.params, 'batch_stats': state.batch_stats}, X,
                            method=lambda module, x: module.features(x, train=False))
    Q_2, norm = Q2(size_p_class = m)
    
    return (H.T@Q_2)/norm


@jax.jit
def invariant(state, x_by_class, kappa_d, kappa_n, kappa_c):
    m = x_by_class[0].shape[0]
    
    H_1 = H1(state, x_by_class)
    H_2 = H2(state, x_by_class)
    
    h_mean = jnp.mean(H_1, axis = 1).reshape(-1,1)
    
    W = weights(state.params)
    
    lambda_0 = kappa_d - kappa_c
    lambda_c = kappa_d - kappa_c + m*(kappa_c - kappa_n)
    
    alpha = kappa_n/(lambda_0/m + kappa_c + (NUM_CLASSES-1)*kappa_n)
    
    return lambda_c*W.T@W - (H_1@H_1.T)*m - lambda_c*(H_2@H_2.T)/lambda_0 + alpha*(NUM_CLASSES**2)*m*h_mean@h_mean.T, lambda_c
    
@jax.jit
def invariant_eot(state, batch, kappa_d, kappa_n, kappa_c):
    
    H = features(state.params, state.batch_stats, batch['image']).T
    W = weights(state.params)
    
    lambda_0 = kappa_d - kappa_c

    return W.T@W - (H@H.T)/lambda_0
    
    
@jax.jit
def NC1(state, x_by_class):
    def h_class_std(x):
        return jnp.std(model.apply({"params": state.params, 'batch_stats': state.batch_stats}, x,
                            method=lambda module, x: module.features(x, train=False)), axis=0)

    return jnp.mean(jax.vmap(h_class_std)(x_by_class))

@jax.jit
def NC2(H_1):
    
    M = H_1 - jnp.mean(H_1, axis = 1).reshape(-1,1)
    prod = M.T@M
    prod = prod/jnp.linalg.norm(prod)
    
    ETF = (jnp.eye(NUM_CLASSES) - jnp.ones((NUM_CLASSES,NUM_CLASSES))/NUM_CLASSES)/jnp.sqrt(NUM_CLASSES-1)
    
    return jnp.linalg.norm(prod - ETF)


@jax.jit
def NC3(W,H_1):
    
    M = H_1 - jnp.mean(H_1, axis = 1).reshape(-1,1)
    return jnp.linalg.norm(M/jnp.linalg.norm(M) - W.T/jnp.linalg.norm(W.T))


@jax.jit
def contract(x,y):
    return jnp.sum(x*y)

# the NTK on a single pair of samples (x1,x2)
def K(state, c):
    @jax.jit
    def K(x1,x2): 
        f = lambda p, x: state.apply_fn({"params": p,
                             'batch_stats': state.batch_stats},x[jnp.newaxis,:],train=False).flatten()[c]


        g1 = jax.grad(lambda p: f(p,x1))(state.params)
        g2 = jax.grad(lambda p: f(p,x2))(state.params)

        return jax.tree_util.tree_reduce(jnp.add,jax.tree_map(contract,g1,g2))
    
    return K

# the NTK matrix (vectorization of K)
def K_matr(state,c):
    _K = K(state,c)
    
    @jax.jit
    def K_matr(X,Y):
        f = lambda x1,x2: _K(x1,x2)
        return jax.vmap(jax.vmap(f,(None,0)),(0,None))(X,Y)
    
    return K_matr


def K_diag(state,c):
    _K = K(state,c)
    
    @jax.jit
    def K_matr(X):
        f = lambda x: _K(x,x)
        return jax.vmap(f,0)(X)
    
    return K_matr

# the NTK_h on a single pair of samples (x1,x2)
def K_h(state, c):

    @jax.jit
    def K(x1,x2): 
        f = lambda p, x: features(p, state.batch_stats, x[jnp.newaxis,:]).flatten()[c]
        g1 = jax.grad(lambda p: f(p,x1))(state.params)
        g2 = jax.grad(lambda p: f(p,x2))(state.params)

        return jax.tree_util.tree_reduce(jnp.add,jax.tree_map(contract,g1,g2))
    
    return K

# the NTK_h matrix (vectorization of K_h)
def K_h_matr(state,c):
    _K = K_h(state,c)
    
    @jax.jit
    def K_matr(X,Y):
        f = lambda x1,x2: _K(x1,x2)
        return jax.vmap(jax.vmap(f,(None,0)),(0,None))(X,Y)
    
    return K_matr

def K_h_diag(state,c):
    _K = K_h(state,c)
    
    @jax.jit
    def K_matr(X):
        f = lambda x: _K(x,x)
        return jax.vmap(f,0)(X)
    
    return K_matr


@jax.jit
def compute_metrics(state, batch, x_by_class, sample_from_batch):
    
    predictions = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            batch['image'],train=False)
    
    if LOSS == 'CE':
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=predictions, labels=batch['label']).mean()
    elif LOSS == 'MSE':
        loss = mse_loss(predictions, batch['label'])

    accuracy = jnp.mean(jnp.argmax(predictions, -1) == batch['label'])
    
    k = int(sample_from_batch.shape[1]/2)
    
    def av_over_c(res, c, len_c, func_to_average, arg):
        f = lambda r,x: func_to_average(r,x,c=c)
        _, ker_vals = lax.scan(f, 0, arg)
        return 0, ker_vals 

    def kappa_d_func_c(res, x, c):
        ker_vals = K_h_diag(state,c)(x)
        return 0, ker_vals
    
    def kappa_c_func_c(res, x, c):
        ker_vals = K_h_matr(state,c)(x[:k],x[k:])
        return 0, ker_vals
    
    def kappa_n_func_c(res, ij, c):
        i = ij[0]
        j = ij[1]
        ker_vals = K_h_matr(state,c)(sample_from_batch[i][:k],sample_from_batch[j][k:])
        return 0, ker_vals
    
    def gamma_d_func_c(res, x, c):
        ker_vals = K_diag(state,c)(x)
        return 0, ker_vals
    
    def gamma_c_func_c(res, x, c):
        ker_vals = K_matr(state,c)(x[:k],x[k:])
        return 0, ker_vals
    
    def gamma_n_func_c(res, ij, c):
        i = ij[0]
        j = ij[1]
        ker_vals = K_matr(state,c)(sample_from_batch[i][:k],sample_from_batch[j][k:])
        return 0, ker_vals

    kappa_d_func = lambda res, c: av_over_c(res, c, NUM_H, kappa_d_func_c, sample_from_batch)
    kappa_c_func = lambda res, c: av_over_c(res, c, NUM_H, kappa_c_func_c, sample_from_batch)
    kappa_n_func = lambda res, c: av_over_c(res, c, NUM_H, kappa_n_func_c, comb_list)

    gamma_d_func = lambda res, c: av_over_c(res, c, NUM_CLASSES, gamma_d_func_c, sample_from_batch)
    gamma_c_func = lambda res, c: av_over_c(res, c, NUM_CLASSES, gamma_c_func_c, sample_from_batch)
    gamma_n_func = lambda res, c: av_over_c(res, c, NUM_CLASSES, gamma_n_func_c, comb_list)
    
    
    _, gamma_d_vals = lax.scan(gamma_d_func, 0, jnp.arange(NUM_CLASSES))
    _, gamma_c_vals = lax.scan(gamma_c_func, 0, jnp.arange(NUM_CLASSES))
    _, gamma_n_vals = lax.scan(gamma_n_func, 0, jnp.arange(NUM_CLASSES))
    
    _, kappa_d_vals = lax.scan(kappa_d_func, 0, jnp.arange(NUM_H))
    _, kappa_c_vals = lax.scan(kappa_c_func, 0, jnp.arange(NUM_H))
    _, kappa_n_vals = lax.scan(kappa_n_func, 0, jnp.arange(NUM_H))

    
    gamma_d = jnp.mean(gamma_d_vals)
    gamma_c = jnp.mean(gamma_c_vals)
    gamma_n = jnp.mean(gamma_n_vals)

    kappa_d = jnp.mean(kappa_d_vals)
    kappa_c = jnp.mean(kappa_c_vals)
    kappa_n = jnp.mean(kappa_n_vals)
    
    H_1 = H1(state, x_by_class)
    H_2 = H2(state, x_by_class)

    h_mean = jnp.mean(H_1, axis = 1).reshape(-1,1)
    W = weights(state.params)

    m = x_by_class[0].shape[0]    
    lambda_0 = kappa_d - kappa_c
    lambda_c = kappa_d - kappa_c + m*(kappa_c - kappa_n)
    lambda_max = kappa_d - kappa_c + m*(kappa_c - kappa_n) + m*NUM_CLASSES*kappa_n

    alpha = kappa_n/(lambda_0/m + kappa_c + (NUM_CLASSES-1)*kappa_n)

    inv_lhs = W.T@W
    inv_rhs = (H_1@H_1.T)*m/lambda_c + (H_2@H_2.T)/lambda_0 - alpha*(NUM_CLASSES**2)*m*h_mean@h_mean.T/lambda_c
    inv = inv_lhs - inv_rhs

    NC1_score = NC1(state, x_by_class)
    NC2_score = NC2(H_1)
    NC3_score = NC3(W,H_1)

    W_norm = jnp.linalg.norm(inv_lhs)
    inv_lhs_normed = inv_lhs/W_norm
    inv_rhs_norm = jnp.linalg.norm(inv_rhs)
    inv_rhs_normed = inv_rhs/inv_rhs_norm 
    inv_frob_dist = 1. - jnp.trace(inv_lhs_normed.T@inv_rhs_normed)
    
    Y_norm = jnp.sqrt(k*k*NUM_CLASSES)
    kappa_d_vals_scaled = jnp.sum(kappa_d_vals,axis=0)/kappa_d
    kappa_c_vals_scaled = jnp.sum(kappa_c_vals,axis=0)/kappa_d
    kappa_n_vals_scaled = jnp.sum(kappa_n_vals,axis=0)/kappa_d
    kappa_norm_scaled = jnp.sqrt(jnp.mean(kappa_d_vals_scaled**2)*k*NUM_CLASSES 
                                 + jnp.mean(kappa_c_vals_scaled**2)*k*(k-1)*NUM_CLASSES  
                                 + jnp.mean(kappa_n_vals_scaled**2)*k*k*NUM_CLASSES*(NUM_CLASSES-1))
    tr_kappa_Y_scaled = jnp.mean(kappa_d_vals_scaled)*k*NUM_CLASSES + jnp.mean(kappa_c_vals_scaled)*k*(k-1)*NUM_CLASSES
    
    kappa_dist_to_Y = 1. - tr_kappa_Y_scaled/kappa_norm_scaled/Y_norm
    
    gamma_d_vals_scaled = jnp.sum(gamma_d_vals,axis=0)/gamma_d
    gamma_c_vals_scaled = jnp.sum(gamma_c_vals,axis=0)/gamma_d
    gamma_n_vals_scaled = jnp.sum(gamma_n_vals,axis=0)/gamma_d
    gamma_norm_scaled = jnp.sqrt(jnp.mean(gamma_d_vals_scaled**2)*k*NUM_CLASSES 
                                 + jnp.mean(gamma_c_vals_scaled**2)*k*(k-1)*NUM_CLASSES  
                                 + jnp.mean(gamma_n_vals_scaled**2)*k*k*NUM_CLASSES*(NUM_CLASSES-1))
    tr_gamma_Y_scaled = jnp.mean(gamma_d_vals_scaled)*k*NUM_CLASSES + jnp.mean(gamma_c_vals_scaled)*k*(k-1)*NUM_CLASSES
    
    gamma_dist_to_Y = 1. - tr_gamma_Y_scaled/gamma_norm_scaled/Y_norm

    metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'gamma_dist_to_Y': gamma_dist_to_Y,
      'kappa_dist_to_Y': kappa_dist_to_Y,
      'alpha': alpha,
      'invariant_norm': jnp.linalg.norm(inv),
      'invariant_alignment': 1. - inv_frob_dist,
      'H1_norm': jnp.linalg.norm(H_1@H_1.T),
      'H2_norm': jnp.linalg.norm(H_2@H_2.T),  
      'global_mean_norm': jnp.linalg.norm(h_mean@h_mean.T),
      'NC1': NC1_score,
      'NC2': NC2_score,
      'NC3': NC3_score
      }

    return metrics



batches = list(test_ds.as_numpy_iterator())
x_batches = jnp.array([x['image'] for x in batches])
y_batches = jnp.array([x['label'] for x in batches])

@jax.jit
def compute_metrics_test(state):

    def compute_batch(carry, batch):
        x, y = batch
        predictions = state.apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats},
                x,train=False)

        if LOSS == 'CE':
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=predictions, labels=y).mean()
        elif LOSS == 'MSE':
            loss = mse_loss(predictions, y)

        accuracy = jnp.mean(jnp.argmax(predictions, -1) == y)

        return 0, [loss, accuracy]

    _, (loss_batches, acc_batches) = lax.scan(compute_batch,0,(x_batches,y_batches))
    
    metrics = {
      'test_loss': jnp.mean(loss_batches),
      'test_accuracy': jnp.mean(acc_batches)
    }
    
    return metrics


model = ResNet(num_classes=NUM_CLASSES, num_blocks=(3, 3, 3), c_hidden=(16, 32, 64))

rng = random.PRNGKey(0)

num_steps_per_epoch = DATASET_TOTAL_SAMPLES_PER_CLASS*NUM_CLASSES/batch_size

LR_decay_func = optax.exponential_decay(init_value=LEARNING_RATE, transition_steps=120*num_steps_per_epoch, decay_rate=0.1, staircase=True)

state = create_train_state(rng, model, lr=LR_decay_func)

metrics_history = {'loss': [],
                   'accuracy': [],
                   'test_loss': [],
                  'test_accuracy': [],
                  'gamma_dist_to_Y': [],
                  'kappa_dist_to_Y': [],
                  'alpha': [],
                  'invariant_norm': [],
                  'invariant_alignment': [],
                  'H1_norm': [],
                  'H2_norm': [],
                  'global_mean_norm': [],
                  'NC1': [],
                  'NC2': [],
                  'NC3': []
                  }

comb_list = jnp.array(list(itertools.combinations(range(NUM_CLASSES), 2)))

for epoch in tqdm(range(num_epochs)):
    
    epoch_ds = train_ds.shuffle(1024, seed=epoch) 
    epoch_ds = tf.data.Dataset.range(NUM_CLASSES).interleave(lambda c: 
                                                         epoch_ds.filter(lambda x: x['label'] == c).batch(DATASET_TOTAL_SAMPLES_PER_CLASS, drop_remainder=True).unbatch(), 
                                                                              cycle_length=NUM_CLASSES)
    epoch_ds = epoch_ds.batch(batch_size, drop_remainder=True).cache().prefetch(10) 


    for step,batch in enumerate(epoch_ds.as_numpy_iterator()):
        state,predictions = train_step(state, batch) 

    if epoch % EPOCH_STEP == 0: 

        x_by_class = jnp.array([batch['image'][jnp.where(batch['label']==c)] for c in range(NUM_CLASSES)])
        sample_from_batch = jnp.array([x_by_class[c][:m] for c in range(NUM_CLASSES)])
        
        metrics = compute_metrics(state, batch, x_by_class, sample_from_batch)
        metrics_test = compute_metrics_test(state)

        for metric, val in metrics.items(): 
            metrics_history[metric].append(val) 

        for metric, val in metrics_test.items(): 
            metrics_history[metric].append(val)

        print(f"train epoch: {epoch+1}, "
              f"loss: {metrics_history['loss'][-1]}, "
              f"accuracy: {metrics_history['accuracy'][-1] * 100}")

        print(f"test loss: {metrics_history['test_loss'][-1]}, "
              f"test_accuracy: {metrics_history['test_accuracy'][-1] * 100}")
        
        pickle.dump(metrics_history, open( PATH+FILE, "wb" ) )


    gc.collect()














