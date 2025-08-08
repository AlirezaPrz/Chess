import keras
import tensorflow as tf

model = keras.models.load_model("hard_dqn_model.h5", compile=False)
tf.keras.models.save_model(model, "hard_dqn_model_tf.h5")       # HDF5

model = keras.models.load_model("medium_dqn_model.h5", compile=False)
tf.keras.models.save_model(model, "meduim_dqn_model_tf.h5")       # HDF5

model = keras.models.load_model("easy_dqn_model.h5", compile=False)
tf.keras.models.save_model(model, "easy_dqn_model_tf.h5")       # HDF5
