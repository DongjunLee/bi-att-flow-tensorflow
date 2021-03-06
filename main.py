#-- coding: utf-8 -*-

import argparse
import logging

from hbconfig import Config
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import data_loader
from model import Model



def experiment_fn(run_config, params):

    model = Model()
    estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)

    train_data, test_data = data_loader.make_train_and_test_set()
    train_X, train_y = train_data
    test_X, test_y = train_data

    train_input_fn, train_input_hook = data_loader.make_batch(train_X, train_y,
                                                              batch_size=Config.model.batch_size,
                                                              scope="train")
    test_input_fn, test_input_hook = data_loader.make_batch(test_X, test_y,
                                                            batch_size=Config.model.batch_size,
                                                            scope="test")

    train_hooks = [train_input_hook]
    if Config.train.print_verbose:
        train_hooks.append(hook.print_variables(
            variables=['train/input_0', 'train/target_0'],
            every_n_iter=Config.train.check_hook_n_iter))
    if Config.train.debug:
        train_hooks.append(tf_debug.LocalCLIDebugHook())

    eval_hooks = [test_input_hook]
    if Config.train.debug:
        eval_hooks.append(tf_debug.LocalCLIDebugHook())

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=train_hooks,
        eval_hooks=eval_hooks
    )
    return experiment


def main(mode):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    run_config = tf.contrib.learn.RunConfig(
            model_dir=Config.train.model_dir,
            save_checkpoints_steps=Config.train.save_checkpoints_steps)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=mode,
        hparams=params
    )

    tf.logging._logger.setLevel(logging.INFO)

    Config(args.config)
    print("Config: ", Config)
    if Config.description:
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main(args.mode)
