#encode=utf-8

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from dnn.config import Config
from dnn.dnn_model import DnnModel

Colors = ['r', 'g', 'b', 'y', 'black']

class MetaData:
    def __init__(self,
                 metric_name,
                 learning_rate = 0.001,
                 mini_batch = True,
                 activation_function = 'relu',
                 optimizer = 'adam',
                 num_iterations = 30,
                 sample=30):
        self.metric_name = metric_name
        self.learning_rate = learning_rate
        self.mini_batch = mini_batch
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.num_iterations = num_iterations
        self.sample = sample
        
    def dprint(self):
        print 'metric_name = ' + self.metric_name
        print 'learning_rate = ' + str(self.learning_rate)
        print 'mini_batch = ' + str(self.mini_batch)
        print 'activation_function = ' + str(self.activation_function)
        print 'optimizer = ' + str(self.optimizer)
        print 'num_iterations = ' + str(self.num_iterations)
    
    def update_config(self):
        Config().mini_batch.mini_batch = self.mini_batch
        Config().learning_rate = self.learning_rate
        Config().layer.activation_function = self.activation_function
        Config().optimizer = self.optimizer
        Config().num_iterations = self.num_iterations
        
    def get_x_label(self):
        return 'epochs = %d, sample = %d'\
             % (self.num_iterations, self.sample)
        
    def get_label(self, run_time):
        return "%s, %.2f second"\
            % (self.metric_name, run_time)

class MetricMetaData:
    def __init__(self, cost, accuracy):
        self.cost = cost
        self.accuracy = accuracy
class Metric:
    def add_train_metric(self, cost, accuracy):
        self.train_metric = MetricMetaData(cost, accuracy)
    def add_test_metric(self, cost, accuracy):
        self.test_metric = MetricMetaData(cost, accuracy)
    def add_validation_metric(self, cost, accuracy):
        self.validation_metric = MetricMetaData(cost, accuracy)

class Benchmark:
    @staticmethod
    def run(meta_data):
        np.random.seed(20171022)
        print '-' * 40
        metric = Metric()
        meta_data.dprint()
        meta_data.update_config()
        start_time = time.time()
        dnn_model = DnnModel()
        cost, accuracy = dnn_model.train(sample = meta_data.sample)
        metric.add_train_metric(cost, accuracy)
        end_time = time.time()
        cost, accuracy = dnn_model.validation()
        metric.add_validation_metric(cost, accuracy)
        cost, accuracy = dnn_model.test()
        metric.add_test_metric(cost, accuracy)
        return (metric, end_time - start_time)
    
    @staticmethod
    def show(folder, metric_name, meta_data, ylabel, loc):
        xlabel = meta_data.get_x_label()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc = loc)
        plt.title(metric_name)
        #plt.show()
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img', folder)
        img_path = os.path.join(folder, metric_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(img_path)
        plt.clf()
        
    @staticmethod
    def predict_wrapper(folder, meta_datas):
        metrics = []
        run_times = []
        for meta_data in meta_datas:
            metric, run_time = Benchmark.run(meta_data)
            metrics.append(metric)
            run_times.append(run_time)
        # print cost
        for meta_data, metric, color, run_time in zip(meta_datas, metrics, Colors, run_times):
            label = meta_data.get_label(run_time)
            plt.plot(np.squeeze(metric.train_metric.cost), color, label = label)
        Benchmark.show(folder, 'cost', meta_data, 'cost (log_likelihold)', 'best')
        # print accuracy
        for meta_data, metric, color, run_time in zip(meta_datas, metrics, Colors, run_times):
            label = meta_data.get_label(run_time)
            plt.plot(np.squeeze(metric.train_metric.accuracy), color, label = label)
            # | ${metric_value} | epochs | run time(seconds) | train accuracy | validation accuracy | test accuracy |
            print '| %s | %d | %.2f | %.2f%% | %.2f%% | %.2f%% |'\
                % (meta_data.metric_name.split(' = ')[-1],
                   meta_data.num_iterations,
                   run_time,
                   metric.train_metric.accuracy[-1],
                   metric.validation_metric.accuracy,
                   metric.test_metric.accuracy
                )
        Benchmark.show(folder, 'accuracy', meta_data, 'accuracy/%', 'best')
        