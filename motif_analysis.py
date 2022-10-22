import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import preprocess.preprocess as preprocess


def seq_only_motif_finding(model, X, motif_size, return_type='motif'):
    last_conv_layer_name = list(filter(lambda x: isinstance(x, keras.layers.Conv2D), model.layers))[-1].name
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    layers = list(filter(lambda x: not isinstance(x, keras.layers.Conv2D), model.layers))
    layer_names = [l.name for l in layers]
    x = classifier_input
    for layer_name in layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    with tf.GradientTape() as tape:
        inputs = X[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    if return_type == 'motif':
        rwa = pd.Series(gradcam[:, 0]).rolling(motif_size).mean()
        return X[rwa.idxmax() - motif_size + 1: rwa.idxmax()+1]
    elif return_type == 'cam-vec':
        return gradcam
    else:
        return None


def save_motif_fasta_files(model, X, motif_size, file_name):
    res = np.zeros((len(X), motif_size, X.shape[2]))
    for i in range(len(X)):
        res[i] = seq_only_motif_finding(model, X[i], motif_size)[:, :, 0]
    f = open(file_name, "w")
    try:
        for i in range(len(res)):
            seq = preprocess.convert_onehot_to_seq(res[i])
            if len(seq) == int(motif_size):
                f.write('>'+str(i)+'\n')
                f.write(seq)
                f.write("\n")
    except:
        print('Something went wrong', seq, i)
    finally:
        f.close()
