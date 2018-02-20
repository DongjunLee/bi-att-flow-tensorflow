# BiDAF (bi-att-flow) [![hb-research](https://img.shields.io/badge/hb--research-experiment-green.svg?style=flat&colorA=448C57&colorB=555555)](https://github.com/hb-research)

TensorFlow implementation of [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) with [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) Task.


## Requirements

- Python 3.6
- TensorFlow 1.4
- [hb-config](https://github.com/hb-research/hb-config) (Singleton Config)
- nltk (tokenizer and blue score)
- tqdm (progress bar)


## Project Structure

init Project by [hb-base](https://github.com/hb-research/hb-base)

    .
    ├── config                  # Config files (.yml, .json) using with hb-config
    ├── data                    # dataset path
    ├── notebooks               # Prototyping with numpy or tf.interactivesession
    ├── BiDAF                   # bi-att-flow architecture graphs (from input to logits)
        ├── __init__.py             # Graph logic
        ├── attention.py            # Attention (Query2Context, Context2Query)
        ├── embedding.py            # Word Embedding, Character Embedding
        └── layer.py                # RNN and other modules
    ├── data_loader.py          # raw_date -> precossed_data -> generate_batch (using Dataset)
    ├── hook.py                 # training or test hook feature (eg. print_variables)
    ├── main.py                 # define experiment_fn
    └── model.py                # define EstimatorSpec      

Reference : [hb-config](https://github.com/hb-research/hb-config), [Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator), [experiments_fn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment), [EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec)

## Todo

- Implements Multi-Head Masked opt
- Train and evaluate with 'WMT German-English (2016)' dataset

## Config

Can control all **Experimental environment**.

example: check-tiny.yml

```yml
data:
  base_path: 'data/'

model:
  batch_size: 4

train:
  learning_rate: 0.0001
  optimizer: 'Adam'  ('Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD')
  
  train_steps: 15000
  model_dir: 'logs/check_tiny'
  
  save_checkpoints_steps: 1000
  check_hook_n_iter: 100
  min_eval_frequency: 100
  
  print_verbose: True
  debug: False
  
```

* debug mode : using [tfdbg](https://www.tensorflow.org/programmers_guide/debugger)
* `check-tiny` is a data set with about **30 sentences** that are translated from Korean into English. (recommend read it :) )

## Usage

Install requirements.

```pip install -r requirements.txt```

Then, pre-process raw data.

```python data_loader.py --config check-tiny```

Finally, start train and evaluate model

```python main.py --config check-tiny --mode train_and_evaluate```


### Experiments modes

:white_check_mark: : Working  
:white_medium_small_square: : Not tested yet.


- : white_medium_small_square: `evaluate` : Evaluate on the evaluation data.
- :white_medium_small_square: `extend_train_hooks` :  Extends the hooks for training.
- :white_medium_small_square: `reset_export_strategies` : Resets the export strategies with the new_export_strategies.
- :white_medium_small_square: `run_std_server` : Starts a TensorFlow server and joins the serving thread.
- :white_medium_small_square: `test` : Tests training, evaluating and exporting the estimator for a single step.
- : white_medium_small_square: `train` : Fit the estimator using the training data.
- : white_medium_small_square: `train_and_evaluate` : Interleaves training and evaluation.

---

### Tensorboar

```tensorboard --logdir logs```


## Reference

- [Paper - Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) (2016. 11) by Minjoon Seo)

## Author

Dongjun Lee (humanbrain.djlee@gmail.com)
