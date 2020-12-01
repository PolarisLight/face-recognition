import tqdm

from utils import *
from VGG16 import *

num_epoch = 5
learning_rate = 1e-5

eps = 1e-5

model = tf.keras.applications.VGG16(input_shape=(256, 256, 3), include_top=True, weights=None, classes=128)
"""

l_layer = len(model.layers)

for i in range(l_layer - 1):
    model.layers[i].trainable = True

model = tf.keras.Sequential(model)
Flatten_layer = tf.keras.layers.Flatten()
new_output = tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=tf.initializers.Constant(0.001))
Desen1 = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
Desen2 = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
model.add(Flatten_layer)
model.add(Desen1)
model.add(Desen2)
model.add(new_output)

model = VGG16(input_shape=(None, 256, 256, 3), output_size=128)
"""
loader = FaceLoader("face\\")

model.build(input_shape=(None, 256, 256, 3))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for e in range(num_epoch):
    if e == 3:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    with tqdm.tqdm(total=1000) as pbar:
        for i in range(1000):
            with tf.GradientTape() as tape:
                src, same, diff = loader.getData()
                src_pred = model(src)
                same_pred = model(same)
                diff_pred = model(diff)

                src_pred = 100 * tf.reshape(src_pred, -1)
                same_pred = 100 * tf.reshape(same_pred, -1)
                diff_pred = 100 * tf.reshape(diff_pred, -1)

                # print(src_pred.numpy().shape)
                # print(src_pred.numpy())
                # print(same_pred.numpy())
                # print(diff_pred.numpy())

                same_distance = get_cos_distance(src_pred, same_pred)
                diff_distance = get_cos_distance(src_pred, diff_pred)

                # print(same_distance)
                # print(diff_distance)

                P_s = same_distance / 2 + 0.5
                P_d = diff_distance / 2 + 0.5

                # d_loss_real = utils.celoss_zeros(P_s)
                # d_loss_fake = utils.celoss_ones(P_d)

                d_loss_real = -tf.math.log(P_s + eps)
                d_loss_fake = -tf.math.log(1 - P_d + eps)

                # print(d_loss_real)
                # print(d_loss_fake)

                loss = d_loss_fake + d_loss_real

                pbar.set_description("total loss = %s,d_loss_real = %s,d_loss_fake = %s" % (loss.numpy(),
                                                                                            d_loss_real.numpy(),
                                                                                            d_loss_fake.numpy()))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            pbar.update(1)
