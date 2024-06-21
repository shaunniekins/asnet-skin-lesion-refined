## Changes from the Original Code

1. In `loss.py`: Changed `from keras.backend import binary_crossentropy` to `from tensorflow.keras.backend import binary_crossentropy`.
2. Commented out `model.load_weights('./checkpoint/weights.hdf5')`.
3. Modified `mcp_save` and `mcp_save_best` checkpoint configurations.
4. Updated `read_from_paths` function.
5. Modified optimizer configuration due to deprecation of `decay` parameter:

   ```python
   from tensorflow.keras.optimizers import Adam
   from tensorflow.keras.optimizers.schedules import ExponentialDecay

   initial_learning_rate = 1e-4
   decay_steps = 10000
   decay_rate = 0.9

   lr_schedule = ExponentialDecay(
       initial_learning_rate,
       decay_steps=decay_steps,
       decay_rate=decay_rate,
       staircase=True)

   optimizer = Adam(learning_rate=lr_schedule)

   model.compile(optimizer=optimizer,
                 loss=WBEC(), metrics=['binary_accuracy'])
   ```