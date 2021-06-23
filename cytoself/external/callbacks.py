"""
This code is a modified version of some of the tf.keras.callbacks modules.
"""
import os
import tempfile
import re

import numpy as np

from tensorflow.compat.v1.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.keras.distribute import (
    multi_worker_training_state as training_state,
)
from tensorflow.python.training import checkpoint_management
from tensorflow.python.lib.io import file_io


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    Example:
    ```python
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    model.fit(data, labels, epochs=100, callbacks=[callback],
        validation_data=(val_data, val_labels))
    ```
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        exclude_initial_epochs=0,
    ):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.exclude_initial_epochs = exclude_initial_epochs

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, " "fallback to auto mode.", mode
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        # Modified to make it monitor only the consecutive previous N epochs.
        if epoch > self.exclude_initial_epochs:
            if epoch == self.exclude_initial_epochs + 1 and self.verbose > 0:
                print(
                    f"\nEarlyStopping has started monitoring {self.monitor} from epoch {epoch + 1}"
                )
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.restore_best_weights:
                        if self.verbose > 0:
                            print(
                                "Restoring model weights from the end of the best epoch."
                            )
                        self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Example:
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    Arguments:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced. new_lr = lr *
          factor
        patience: number of epochs with no improvement after which learning rate
          will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
          quantity monitored has stopped decreasing; in `max` mode it will be
          reduced when the quantity monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred from the name of the
          monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
          lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        exclude_initial_epochs=0,
        **kwargs,
    ):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError("ReduceLROnPlateau " "does not support a factor >= 1.0.")
        if "epsilon" in kwargs:
            min_delta = kwargs.pop("epsilon")
            logging.warning(
                "`epsilon` argument is deprecated and "
                "will be removed, use `min_delta` instead."
            )
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.exclude_initial_epochs = exclude_initial_epochs
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ["auto", "min", "max"]:
            logging.warning(
                "Learning Rate Plateau Reducing mode %s is unknown, "
                "fallback to auto mode.",
                self.mode,
            )
            self.mode = "auto"
        if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Reduce LR on plateau conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            # Modified to make it monitor only the consecutive previous N epochs.
            if epoch > self.exclude_initial_epochs:
                if epoch == self.exclude_initial_epochs + 1 and self.verbose > 0:
                    print(
                        f"\nReduceLROnPlateau has started monitoring {self.monitor} from epoch {epoch + 1}"
                    )
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            K.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                print(
                                    "\nEpoch %05d: ReduceLROnPlateau reducing learning "
                                    "rate to %s." % (epoch + 1, new_lr)
                                )
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


# TF1.15 ver
class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, the latest best model according
          to the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
          overwrite the current save file is made based on either the maximization
          or the minimization of the monitored quantity. For `val_acc`, this
          should be `max`, for `val_loss` this should be `min`, etc. In `auto`
          mode, the direction is automatically inferred from the name of the
          monitored quantity.
        save_weights_only: if True, then only the model's weights will be saved
          (`model.save_weights(filepath)`), else the full model is saved
          (`model.save(filepath)`).
        save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
          the model after each epoch. When using integer, the callback saves the
          model at end of a batch at which this many samples have been seen since
          last saving. Note that if the saving isn't aligned to epochs, the
          monitored metric may potentially be less reliable (it could reflect as
          little as 1 batch, since the metrics get reset every epoch). Defaults to
          `'epoch'`
        **kwargs: Additional arguments for backwards compatibility. Possible key
          is `period`.
    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        exclude_initial_epochs=0,
        **kwargs,
    ):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.exclude_initial_epochs = exclude_initial_epochs
        self.epochs_since_last_save = 0
        self._samples_seen_since_last_saving = 0

        # Deprecated field `load_weights_on_restart` is for loading the checkpoint
        # file from `filepath` at the start of `model.fit()`
        # TODO(rchao): Remove the arg during next breaking release.
        if "load_weights_on_restart" in kwargs:
            self.load_weights_on_restart = kwargs["load_weights_on_restart"]
            logging.warning(
                "`load_weights_on_restart` argument is deprecated. "
                "Please use `model.load_weights()` for loading weights "
                "before the start of `model.fit()`."
            )
        else:
            self.load_weights_on_restart = False

        # Deprecated field `period` is for the number of epochs between which
        # the model is saved.
        if "period" in kwargs:
            self.period = kwargs["period"]
            logging.warning(
                "`period` argument is deprecated. Please use `save_freq` "
                "to specify the frequency in number of samples seen."
            )
        else:
            self.period = 1

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "ModelCheckpoint mode %s is unknown, " "fallback to auto mode.", mode
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError("Unrecognized save_freq: {}".format(self.save_freq))

        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False

    def set_model(self, model):
        self.model = model
        # Use name matching rather than `isinstance` to avoid circular dependencies.
        if (
            not self.save_weights_only
            and not model._is_graph_network
            and model.__class__.__name__  # pylint: disable=protected-access
            != "Sequential"
        ):
            self.save_weights_only = True

    def on_train_begin(self, logs=None):
        # pylint: disable=protected-access
        if self.model._in_multi_worker_mode():
            # MultiWorkerTrainingState is used to manage the training state needed
            # for preemption-recovery of a worker in multi-worker training.
            self.model._training_state = training_state.MultiWorkerTrainingState(
                self.model, self.filepath
            )
            self._training_state = self.model._training_state
            if self._training_state.restore():
                # If the training state needs to be and is successfully restored,
                # it is recovering from a previous failure (or preemption). In such
                # case, do not load the weights from user specified file path.
                return

        # If this is not multi worker training, restoring is not needed, or
        # restoring failed, check if it should load weights on restart.
        if self.load_weights_on_restart:
            if (
                not self.model._in_multi_worker_mode()
                or multi_worker_util.should_load_checkpoint()
            ):
                filepath_to_load = self._get_most_recently_modified_file_matching_pattern(
                    self.filepath
                )
                if filepath_to_load is not None and training_state.checkpoint_exists(
                    filepath_to_load
                ):
                    try:
                        # `filepath` may contain placeholders such as `{epoch:02d}`, and
                        # thus it attempts to load the most recently modified file with file
                        # name matching the pattern.
                        self.model.load_weights(filepath_to_load)
                    except (IOError, ValueError) as e:
                        raise ValueError(
                            "Error loading file from {}. Reason: {}".format(
                                filepath_to_load, e
                            )
                        )

    def on_train_end(self, logs=None):
        # pylint: disable=protected-access
        if self.model._in_multi_worker_mode():
            # In multi-worker training, on successful exit of training, delete the
            # training state backup file that was saved for the purpose of worker
            # recovery.
            self._training_state.delete_backup()
            # Restore the training state so the model is ready for next (possible)
            # multi worker training.
            del self._training_state
            del self.model._training_state

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if isinstance(self.save_freq, int):
            self._samples_seen_since_last_saving += logs.get("size", 1)
            if self._samples_seen_since_last_saving >= self.save_freq:
                self._save_model(epoch=self._current_epoch, logs=logs)
                self._samples_seen_since_last_saving = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == "epoch":
            if self.model._in_multi_worker_mode():
                # Exclude training state variables in user-requested checkpoint file.
                with self._training_state.untrack_vars():
                    self._save_model(epoch=epoch, logs=logs)
            else:
                self._save_model(epoch=epoch, logs=logs)
        if self.model._in_multi_worker_mode():
            # For multi-worker training, back up the weights and current training
            # state for possible future recovery.
            # TODO(rchao): Call `back_up` at finer period such as N steps.
            self._training_state.back_up(epoch)

    def _save_model(self, epoch, logs):
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            if self.save_best_only:

                # Modified to make it monitor only the consecutive previous N epochs.
                if epoch > self.exclude_initial_epochs:
                    if epoch == self.exclude_initial_epochs + 1 and self.verbose > 0:
                        print(
                            f"\nModelCheckpoint has started monitoring {self.monitor} from epoch {epoch + 1}"
                        )

                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning(
                            "Can save best model only with %s available, " "skipping.",
                            self.monitor,
                        )
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print(
                                    "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                    " saving model to %s"
                                    % (
                                        epoch + 1,
                                        self.monitor,
                                        self.best,
                                        current,
                                        filepath,
                                    )
                                )
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print(
                                    "\nEpoch %05d: %s did not improve from %0.5f"
                                    % (epoch + 1, self.monitor, self.best)
                                )
            else:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

            self._maybe_remove_file()

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        if (
            not self.model._in_multi_worker_mode()
            or multi_worker_util.should_save_checkpoint()
        ):
            return self.filepath.format(epoch=epoch + 1, **logs)
        else:
            # If this is multi-worker training, and this worker should not
            # save checkpoint, we use a temp filepath to store a dummy checkpoint, so
            # it writes to a file that will be removed at the end of `_save_model()`
            # call. This is because the SyncOnReadVariable needs to be synced across
            # all the workers in order to be read, and all workers need to initiate
            # that.
            self._temp_file_dir = tempfile.mkdtemp()
            extension = os.path.splitext(self.filepath)[1]
            return os.path.join(self._temp_file_dir, "temp" + extension)

    def _maybe_remove_file(self):
        # Remove the checkpoint directory in multi-worker training where this worker
        # should not checkpoint. It is a dummy directory previously saved for sync
        # distributed training.

        if (
            self.model._in_multi_worker_mode()
            and not multi_worker_util.should_save_checkpoint()  # pylint: disable=protected-access
        ):
            file_io.delete_recursively(self._temp_file_dir)
            del self._temp_file_dir

    def _get_most_recently_modified_file_matching_pattern(self, pattern):
        """Returns the most recently modified filepath matching pattern.
        Pattern may contain python formatting placeholder. If
        `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
        check for most recently modified one that matches the pattern.
        In the rare case where there are more than one pattern-matching file having
        the same modified time that is most recent among all, return the filepath
        that is largest (by `>` operator, lexicographically using the numeric
        equivalents). This provides a tie-breaker when multiple files are most
        recent. Note that a larger `filepath` can sometimes indicate a later time of
        modification (for instance, when epoch/batch is used as formatting option),
        but not necessarily (when accuracy or loss is used). The tie-breaker is
        put in the logic as best effort to return the most recent, and to avoid
        undeterministic result.
        Modified time of a file is obtained with `os.path.getmtime()`.
        This utility function is best demonstrated via an example:
        ```python
        file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        file_paths = [
            os.path.join(test_dir, file_name) for file_name in
            ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
        ]
        for file_path in file_paths:
          # Write something to each of the files
        self.assertEqual(
            _get_most_recently_modified_file_matching_pattern(path_pattern),
            file_paths[-1])
        ```
        Arguments:
            pattern: The file pattern that may optionally contain python placeholder
                such as `{epoch:02d}`.
        Returns:
            The most recently modified file's full filepath matching `pattern`. If
            `pattern` does not contain any placeholder, this returns the filepath
            that
            exactly matches `pattern`. Returns `None` if no match is found.
        """
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = "^" + re.sub(r"{.*}", r".*", base_name) + "$"

        # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
        # use that as it is more robust than `os.path.getmtime()`.
        latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(
            base_name_regex, os.path.basename(latest_tf_checkpoint)
        ):
            return latest_tf_checkpoint

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if file_io.file_exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (
                        file_path_with_largest_file_name is None
                        or file_path > file_path_with_largest_file_name
                    ):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found, reset
                        # the counter for the number of files with latest modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the most recent,
                        # increment the counter for the number of files with latest modified
                        # time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time, return
            # the file path with the largest file name.
            return file_path_with_largest_file_name
