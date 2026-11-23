"""Pérdida de destilación por logits (Hinton et al., 2015)."""
import tensorflow as tf
from tensorflow import keras

ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
kl = keras.losses.KLDivergence()


def perdida_kd(y, logits_alum, logits_prof, alpha, T):
    """L = alpha * CE(y, s) + (1-alpha) * T^2 * KL( softmax(t/T) || softmax(s/T) )
    - T suaviza las distribuciones y expone la 'dark knowledge' (similitud entre clases).
    - T^2 compensa que los gradientes del término suave escalan como 1/T^2."""
    l_real = ce(y, logits_alum)
    p_t = tf.nn.softmax(logits_prof / T, axis=-1)
    p_s = tf.nn.softmax(logits_alum / T, axis=-1)
    l_kd = kl(p_t, p_s) * (T ** 2)
    return alpha * l_real + (1.0 - alpha) * l_kd, l_real, l_kd
