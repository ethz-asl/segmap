import numpy as np

class Batchmaker:
    def __init__(self, input_data, latent_data, examples_per_batch, model_params, shuffle_examples=True):
        self.input_data = input_data
        self.input_shape = model_params.INPUT_SHAPE
        self.latent_data = latent_data
        self.n_latent_dims_to_target = len(model_params.COERCED_LATENT_DIMS)
        if self.n_latent_dims_to_target > 0:
          if len(latent_data[0]) != self.n_latent_dims_to_target:
            print(latent_data)
            raise ValueError('Unexpected amount of latent targets given.')
        else:
          if latent_data is not None:
            print(latent_data)
            raise ValueError('Expected no latent targets (==None), but was given some.')
          self.latent_data = [[] for _ in input_data]
        # examples per batch
        if examples_per_batch is "max":
            examples_per_batch = len(input_data)
        assert type(examples_per_batch) is int
        if examples_per_batch > len(input_data):
            print("WARNING: more examples per batch than possible examples in all input_data")
            self.examples_per_batch = len(input_data)
        else:
            self.examples_per_batch = examples_per_batch
        # initialize example indices list
        self.remaining_example_indices = list(range(len(input_data)))
        # shuffle list if required
        if shuffle_examples:
            from random import shuffle
            shuffle(self.remaining_example_indices)
        self.batches_consumed_counter = 0

    def next_batch(self):
        assert not self.is_depleted()
        # Create a single batch
        batch_input_values  =  np.zeros([self.examples_per_batch] + self.input_shape)
        batch_target_values = [np.zeros((self.examples_per_batch))
                               for _ in range(self.n_latent_dims_to_target)]
        for i_example in range(self.examples_per_batch):
          # Create training example at index 'pos' in input_data.
          pos = self.remaining_example_indices.pop(0)
          #   input.
          batch_input_values[i_example] = np.reshape(self.input_data[pos], self.input_shape)
          #   target.
          unrolled_target_data = self.latent_data[pos]
          for t, value in enumerate(unrolled_target_data):
            batch_target_values[t][i_example] = value

        self.batches_consumed_counter += 1

        return batch_input_values, batch_target_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch

    def n_batches_consumed(self):
        return self.batches_consumed_counter

def progress_bar(batchmaker):
  from matplotlib import pyplot as plt  
  import time
  plt.figure('progress_bar')
  plt.scatter(time.time(), batchmaker.n_batches_consumed())
  plt.ylim([0, batchmaker.n_batches_consumed()+batchmaker.n_batches_remaining()])
  plt.show()
  plt.gcf().canvas.draw()
  time.sleep(0.0001)
