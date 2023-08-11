import tensorflow as tf


# 创建一个回调函数来设置阈值
class ThresholdCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, threshold, logs=None):
        self.model.threshold = threshold
