from __future__ import print_function

import os
from collections import defaultdict
from math import sin, pi

import numpy as np
import torch
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from deep_disfluency_utils import get_tag_mapping
from model import AWD_LSTM_DisfluencyDetector


def train(in_model,
          train_data,
          dev_data,
          test_data,
          in_task_vocabs,
          in_model_folder,
          in_epochs_number,
          config,
          class_weights=None,
          task_weights=None,
          **kwargs):
    X_train, y_train_for_tasks = train_data

    tag_mapping = get_tag_mapping(in_task_vocabs[0][1])

    if class_weights is None:
        class_weights = [torch.FloatTensor(np.ones(y_train_i.shape[1])) for y_train_i in y_train_for_tasks]
    else:
        class_weights = [torch.FloatTensor(weights_i) for weights_i in class_weights]
    if torch.cuda.is_available():
        class_weights = [class_weights_i.to(torch.device('cuda')) for class_weights_i in class_weights]
    if task_weights is None:
        task_weights = np.ones(len(y_train_for_tasks))
    # Define loss and optimizer

    starting_lr = config['lr']
    lr_decay = config['lr_decay']

    optimizer = SGD(in_model.parameters(), lr=starting_lr, weight_decay=config['l2_coef'])
    learning_rate_scheduler = CosineAnnealingLR(optimizer, 2000000)

    _, dev_eval = evaluate(in_model,
                           dev_data,
                           tag_mapping,
                           class_weights,
                           task_weights,
                           config)
    best_dev_f1_rm = dev_eval['f1_rm']
    epochs_without_improvement = 0
    for epoch_counter in range(in_epochs_number):
        batch_gen = batch_generator(X_train, y_train_for_tasks, config['batch_size'])
        train_batch_losses = []
        for batch_x, batch_y in batch_gen:
            batch_x = torch.LongTensor(batch_x)
            if torch.cuda.is_available():
                batch_x = batch_x.to(torch.device('cuda'))
            in_model.train()
            batch_pred = in_model(batch_x)
            batch_y_tensors = [torch.LongTensor(batch_y_i) for batch_y_i in batch_y]
            if torch.cuda.is_available():
                batch_y_tensors = [batch_y_i.to(torch.device('cuda')) for batch_y_i in batch_y_tensors]
            train_batch_loss = calculate_loss(batch_pred,
                                              batch_y_tensors,
                                              class_weights,
                                              task_weights=task_weights)
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_batch_losses.append(train_batch_loss)
        _, dev_eval = evaluate(in_model,
                               dev_data,
                               tag_mapping,
                               class_weights,
                               task_weights,
                               config)
        print('Epoch {} out of {} results'.format(epoch_counter, in_epochs_number))
        print('train loss: {:.3f}'.format(torch.mean(torch.Tensor(train_batch_losses)).numpy()))
        print('; '.join(['dev {}: {:.3f}'.format(key, value)
                         for key, value in dev_eval.items()]) + ' @lr={}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        if best_dev_f1_rm < dev_eval['f1_rm']:
            best_dev_f1_rm = dev_eval['f1_rm']
            torch.save(in_model.state_dict(), os.path.join(in_model_folder, AWD_LSTM_DisfluencyDetector.MODEL_FILE))
            print('New best loss. Saving checkpoint')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if config['early_stopping_threshold'] < epochs_without_improvement:
            print('Early stopping after {} epochs'.format(epoch_counter))
            break
        learning_rate_scheduler.step()

    print('Optimization Finished!')


def evaluate(in_model,
             in_dataset,
             in_tag_map,
             in_class_weights,
             in_task_weights,
             in_config,
             batch_size=32):
    in_model.eval()
    X_test, y_test_for_tasks = in_dataset

    batch_gen = batch_generator(X_test, y_test_for_tasks, batch_size)

    batch_losses, batch_accuracies = [], []
    y_pred_main_task = np.zeros(X_test.shape[0])

    for batch_idx, (batch_x, batch_y) in enumerate(batch_gen):
        batch_tensor = torch.LongTensor(batch_x)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.to(torch.device('cuda'))
        y_pred_batch = in_model(batch_tensor)
        batch_y_tensors = [torch.LongTensor(batch_y_i) for batch_y_i in batch_y]
        if torch.cuda.is_available():
            batch_y_tensors = [batch_y_i.to(torch.device('cuda')) for batch_y_i in batch_y_tensors]
        loss_batch = calculate_loss(y_pred_batch,
                                    batch_y_tensors,
                                    in_class_weights,
                                    task_weights=in_task_weights).detach()
        y_predicted = [torch.argmax(logits_i, 1).cpu() for logits_i in y_pred_batch]
        acc_batch = np.mean([y_predicted_i.detach().numpy() == batch_y_i
                             for y_predicted_i, batch_y_i in zip(y_predicted, batch_y)])

        y_pred_main_task[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_predicted[0].detach().numpy()
        batch_losses.append(loss_batch)
        batch_accuracies.append(acc_batch)

    y_gold_main_task = y_test_for_tasks[0]
    result_map = {'loss': torch.mean(torch.Tensor(batch_losses)).numpy(), 'acc': np.mean(batch_accuracies)}
    for class_name, class_ids in in_tag_map.items():
        result_map['f1_' + class_name] = metrics.f1_score(y_true=y_gold_main_task,
                                                          y_pred=y_pred_main_task,
                                                          labels=class_ids,
                                                          average='micro')
    return y_pred_main_task, result_map


def get_sample_weight(in_labels, in_class_weight_map):
    sample_weight = np.vectorize(in_class_weight_map.get)(in_labels)
    return sample_weight


def get_class_weight_sqrt(in_labels):
    label_freqs = defaultdict(lambda: 0)
    for label in in_labels:
        label_freqs[label] += 1.0
    label_weights = {label: 1.0 / np.power(freq, 1 / 3.) for label, freq in label_freqs.items()}
    return label_weights


def get_class_weight_proportional(in_labels, smoothing_coef=1.0):
    label_freqs = defaultdict(lambda: 0)
    for label in in_labels:
        label_freqs[label] += 1.0
    label_weights = {label: 1.0 / np.power(float(freq), 1.0 / smoothing_coef)
                     for label, freq in label_freqs.items()}
    return label_weights


def get_class_weight_auto(in_labels):
    class_weight = compute_class_weight('balanced', np.unique(in_labels), in_labels)
    class_weight_map = {class_id: weight
                        for class_id, weight in zip(np.unique(in_labels), class_weight)}

    return class_weight_map


def random_batch_generator(data, labels, batch_size, sample_probabilities=None):
    data_idx = range(labels.shape[0])
    while True:
        batch_idx = np.random.choice(data_idx, size=batch_size, p=sample_probabilities)
        batch = (np.take(data, batch_idx, axis=0), np.take(labels, batch_idx, axis=0))
        yield batch


def dynamic_importance_sampling_random_batch_generator(data,
                                                       labels,
                                                       batch_size,
                                                       smoothing_coef_min,
                                                       smoothing_coef_max):
    labels_flat = np.argmax(labels, axis=-1)

    sample_probs = np.ones(labels.shape[0])
    data_idx = range(labels.shape[0])
    x = 0.0
    delta = smoothing_coef_min - smoothing_coef_max
    batch_counter = 0
    while True:
        if batch_counter == 0:
            class_weight = get_class_weight_proportional(labels_flat, smoothing_coef=smoothing_coef_min + delta * abs(sin(x)))
            sample_weight = get_sample_weight(labels_flat, class_weight)
            sample_probs = sample_weight / sum(sample_weight)
            x = (x + 0.1) % (2 * pi)
        batch_counter = (batch_counter + 1) % 1000
        batch_idx = np.random.choice(data_idx, size=batch_size, p=sample_probs)
        batch = (np.take(data, batch_idx, axis=0), np.take(labels, batch_idx, axis=0))
        yield batch


def batch_generator(X, y_for_tasks, batch_size):
    batch_start_idx = 0
    total_batches_number = X.shape[0] // batch_size
    for _ in tqdm(range(total_batches_number)):
        batch = (X[batch_start_idx: batch_start_idx + batch_size],
                 [y_i[batch_start_idx: batch_start_idx + batch_size] for y_i in y_for_tasks])
        batch_start_idx += batch_size
        yield batch
        if not (batch_start_idx < X.shape[0]):
            break


def calculate_loss(in_logits_for_tasks,
                   in_labels_for_tasks,
                   in_class_weights_for_tasks,
                   l2_coef=0.0,
                   weight_change_penalization_coef=0.0,
                   task_weights=None):
    assert len(in_logits_for_tasks) == len(in_labels_for_tasks) == len(task_weights)
    if task_weights == None:
        task_weights = np.ones(len(in_logits_for_tasks))

    losses = []
    for logits, labels, class_weights in zip(in_logits_for_tasks,
                                             in_labels_for_tasks,
                                             in_class_weights_for_tasks):
        logits_i = logits
        loss_xent_i = CrossEntropyLoss(weight=class_weights)(logits_i, labels)
        losses.append(loss_xent_i)

    '''
    graph = tf.get_default_graph()
    # L2 regularization on model weights trajectory
    weight_change_l2s = []
    all_tensor_names = [v.name for v in tf.trainable_variables()]
    for v in tf.trainable_variables():
        v_initial_name = v.name.partition(':')[0] + '_initial:' + v.name.partition(':')[-1]
        if v_initial_name not in all_tensor_names:
            continue
        v_initial = graph.get_tensor_by_name(v_initial_name)
        weight_change_l2s.append(tf.nn.l2_loss(tf.subtract(v, v_initial)))
    loss_weight_change = None if not len(weight_change_l2s) \
                         else tf.reduce_sum(weight_change_l2s) * weight_change_penalization_coef
    '''
    # losses_weighted = [loss_i * task_weight_i for loss_i, task_weight_i in zip(losses, task_weights)]
    # aux_losses = [l for l in [loss_l2, loss_weight_change] if l is not None]
    # cost = torch.autograd.Variable(torch.sum(torch.Tensor(losses_weighted)), requires_grad=True)

    return losses[0] * task_weights[0] + losses[1] * task_weights[1]

