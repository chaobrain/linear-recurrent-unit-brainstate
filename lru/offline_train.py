# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from functools import partial
from pprint import pprint

import brainstate as bst
import braintools.file
import jax
import jax.numpy as jnp
import optax
import wandb
from tqdm import tqdm

from .dataloaders import Datasets
from .offline_model import OfflineModel


@jax.vmap
def batched_average_mask(a, mask):
    """Average of a by sum of values of mask"""
    return a / jnp.sum(mask)


@jax.vmap
def create_mask(x, length):
    L = x.shape[0]
    mask = (jnp.arange(L) >= length[0]) * (jnp.arange(L) < length[1])
    return mask


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * logits)


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits) == label


def compute_accuracies(logits, labels, masks):
    if len(logits.shape) == 4:
        return jnp.sum(batched_average_mask(masks * compute_accuracy(logits, labels).mean(axis=-1), masks), axis=-1)
    elif len(logits.shape) == 2:
        return jnp.mean(compute_accuracy(logits, labels))
    else:
        raise RuntimeError("Unhandled shape for logits")


def loss_fn(logits, labels, masks):
    """
    Pick the desired loss depending on the shape of the logits (and therefore the task)
    """
    if len(logits.shape) == 2:  # for classification tasks
        losses = cross_entropy_loss(logits, labels)
    elif len(logits.shape) == 4:  # for tasks with multidimensional dense targets
        losses = masks * cross_entropy_loss(logits, labels).mean(axis=-1)
        losses = batched_average_mask(losses, masks)  # average over time
    else:
        raise RuntimeError("Unhandled shape for logits")
    return jnp.mean(losses)


def prep_batch(batch, seq_len, in_dim):
    """Take a batch and convert it to a standard x/y format"""
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Unhandled data type. ")

    inputs = jnp.array(inputs.numpy()).astype(float)  # convert to jax
    targets = jnp.array(targets.numpy())  # convert to jax
    lengths = aux_data.get("lengths", None)  # get lengths from aux if it is there.

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        inputs = jnp.pad(inputs, ((0, 0), (0, num_pad)), "constant", constant_values=(0,))

    # Inputs size is [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = jax.nn.one_hot(inputs, in_dim)

    if lengths is not None:
        lengths = jnp.array(lengths)
        if len(lengths.shape) == 1:  # If lengths only give last
            lengths = jnp.stack([jnp.zeros((inputs.shape[0],)), lengths], axis=1)
        masks = create_mask(inputs, lengths)
    else:
        masks = jnp.ones((inputs.shape[0], inputs.shape[1]))

    return inputs, targets, masks


def separate_ssm_and_reg(a_dict):
    new_dict = dict()
    for k, v in a_dict.items():
        new_dict[k] = (
            jax.tree.map(lambda a: "ssm", v)
            if k[-1] in ["nu_log", "theta_log", "gamma_log", "B_re", "B_im"] else
            jax.tree.map(lambda a: "regular", v)
        )
    return new_dict


class Trainer:
    def __init__(
        self,
        args,
        dense_targets: bool,
        in_dim: int,
        n_classes: int,
        steps_per_epoch: int,
    ):
        self.args = args

        # ---- model ----
        # ---------------
        model = OfflineModel(
            d_input=in_dim,
            d_hidden=args.d_hidden,
            d_model=args.d_model,
            d_output=n_classes,
            n_layers=args.n_layers,
            r_min=args.r_min,
            r_max=args.r_max,
            dropout=args.p_dropout,
            norm=args.norm,
            multidim=1 + dense_targets,
            pooling=args.pooling,
        )
        self.model = model

        # ---- parameters ----
        # ---------------------
        self.params = bst.graph.states(self.model, bst.ParamState)
        param_size = sum([l.size * 2 if (l.dtype in [jnp.complex64, jnp.complex128]) else l.size
                          for l in jax.tree.leaves(self.params.to_dict_values())])
        print(f"[*] Trainable Parameters: {param_size}")

        # ---- optimizers ----
        # --------------------

        # Smaller lr and no weight decay for lambda, gamma and B
        ssm_lr = bst.ShortTermState(args.lr_factor * args.lr_base)
        reg_lr = bst.ShortTermState(args.lr_base)

        # lr scheduler
        if args.cosine_anneal:
            total_step = steps_per_epoch * args.epochs
            ssm_lr = bst.optim.CosineAnnealingLR(ssm_lr, total_step, self.args.lr_min)
            reg_lr = bst.optim.CosineAnnealingLR(reg_lr, total_step, self.args.lr_min)
        else:
            ssm_lr = bst.optim.ConstantLR(ssm_lr)
            reg_lr = bst.optim.ConstantLR(reg_lr)
        self.ssm_lr = ssm_lr
        self.reg_lr = reg_lr

        # optimizers
        tx = optax.multi_transform(
            {
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=self.ssm_lr.lr),
                "regular": optax.inject_hyperparams(optax.adamw)(
                    learning_rate=self.reg_lr.lr,
                    weight_decay=args.weight_decay
                ),
            },
            separate_ssm_and_reg,
        )

        self.optimizer = bst.optim.OptaxOptimizer(tx)
        self.optimizer.register_trainable_weights(self.params)

    def get_ssm_learning_rate(self):
        return self.optimizer.opt_state.value.inner_states["ssm"].inner_state.hyperparams["learning_rate"]

    def set_ssm_learning_rate(self, lr):
        self.optimizer.opt_state.value.inner_states["ssm"].inner_state.hyperparams["learning_rate"] = lr

    def get_reg_learning_rate(self):
        return self.optimizer.opt_state.value.inner_states["regular"].inner_state.hyperparams["learning_rate"]

    def set_reg_learning_rate(self, lr):
        self.optimizer.opt_state.value.inner_states["regular"].inner_state.hyperparams["learning_rate"] = lr

    @bst.compile.jit(static_argnums=0)
    def train_step(self, inputs, labels, masks):
        """Performs a single training step given a batch of data"""

        @bst.augment.vmap(in_axes=(None, 0, 0), axis_name='batch')
        def run(model, inp, key):
            with bst.environ.context(fit=True):
                bst.random.set_key(key)
                return model(inp)

        def _loss():
            logits = run(self.model, inputs, bst.random.split_key(inputs.shape[0]))
            return loss_fn(logits, labels, masks)

        grads, loss = bst.augment.grad(_loss, self.params, return_value=True)()
        self.optimizer.update(grads)

        # update learning rate per step
        self.reg_lr.step_epoch()
        self.ssm_lr.step_epoch()
        self.set_reg_learning_rate(self.reg_lr())
        self.set_ssm_learning_rate(self.ssm_lr())
        return loss

    def train_epoch(self, trainloader, seq_len, in_dim):
        """
        Training function for an epoch that loops over batches.
        """
        batch_losses = []
        for batch in tqdm(trainloader):
            inputs, labels, masks = prep_batch(batch, seq_len, in_dim)
            loss = self.train_step(inputs, labels, masks)
            batch_losses.append(loss)  # log loss value
        return jnp.mean(jnp.array(batch_losses))

    @bst.compile.jit(static_argnums=0)
    def eval_step(self, inputs, labels, masks):

        @bst.augment.vmap(in_axes=(None, 0, 0), axis_name='batch')
        def run(model, inp, key):
            with bst.environ.context(fit=False):
                bst.random.set_key(key)
                return model(inp)

        logits = run(self.model, inputs, bst.random.split_key(inputs.shape[0]))
        losses = loss_fn(logits, labels, masks)
        accs = compute_accuracies(logits, labels, masks)
        return jnp.mean(losses), accs

    def validate(self, testloader, seq_len, in_dim):
        """Validation function that loops over batches"""
        losses, accuracies = jnp.array([]), jnp.array([])

        for batch in tqdm(testloader):
            inputs, labels, masks = prep_batch(batch, seq_len, in_dim)
            loss, acc = self.eval_step(inputs, labels, masks)
            losses = jnp.append(losses, loss)
            accuracies = jnp.append(accuracies, acc)
        return jnp.mean(losses), jnp.mean(accuracies)

    def reduce_lr_on_plateau(self, count, opt_acc, new_acc):
        factor = self.args.reduce_factor
        patience = self.args.lr_patience
        lr_min = self.args.lr_min

        lr = self.reg_lr.lr
        ssm_lr = self.ssm_lr.lr
        if new_acc > opt_acc:
            count = 0
            opt_acc = new_acc
        else:
            count += 1

        if count > patience:
            lr = factor * lr
            ssm_lr = factor * ssm_lr
            count = 0

        if lr < lr_min:
            lr = lr_min
        if ssm_lr < lr_min:
            ssm_lr = lr_min

        self.reg_lr.lr = lr
        self.ssm_lr.lr = ssm_lr

        return count, opt_acc


def offline_train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    if args.use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=args.wandb_project,
            job_type="model_training",
            config=vars(args),
            entity=args.wandb_entity,
        )
    else:
        wandb.init(mode="offline")

    # Set randomness...
    print("[*] Setting Randomness...")
    bst.random.seed(args.jax_seed)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]
    # Dataset dependent logic
    if args.dataset == "copy-classification":
        assert args.pooling == "none", "No pooling for copy task"
        dense_targets = True
    else:
        dense_targets = False
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        if args.dataset in ["aan-classification"]:
            # Use retrieval model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False
    else:
        retrieval = False

    # Create dataset...
    (
        trainloader,
        valloader,
        testloader,
        aux_dataloaders,
        n_classes,
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset_fn(args.dir_name, seed=args.jax_seed, batch_size=args.batch_size)
    print(f"[*] Starting training on `{args.dataset}` =>> Initializing...")

    if retrieval:
        raise NotImplementedError("Retrieval model not implemented yet")

    # Initialize trainer
    trainer = Trainer(args, dense_targets, in_dim, n_classes, steps_per_epoch=int(train_size / args.batch_size))

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        train_loss = trainer.train_epoch(trainloader, seq_len, in_dim)
        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = trainer.validate(valloader, seq_len, in_dim)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = trainer.validate(testloader, seq_len, in_dim)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} "
                f"-- Val Loss: {val_loss:.5f} "
                f"-- Test Loss: {test_loss:.5f}\n"
                f"\tVal Accuracy: {val_acc:.4f} "
                f"-- Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = trainer.validate(testloader, seq_len, in_dim)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  -- Test Loss: {val_loss:.5f}\n"
                f"\tTest Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            # # Save best model
            # params = trainer.params.to_nest()
            # braintools.file.msgpack_save('best_model.msgpack', params)
            # braintools.file.msgpack_load('best_model.msgpack', params)

        # For learning rate decay purposes:
        lr_count, opt_acc = trainer.reduce_lr_on_plateau(lr_count, opt_acc, val_acc)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        metrics = {
            "Training Loss": train_loss,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Count": count,
            "Learning rate count": lr_count,
            "Opt acc": opt_acc,
            "lr": trainer.reg_lr.lr,
            "ssm_lr": trainer.ssm_lr.lr,
        }
        if valloader is not None:
            metrics["Test Loss"] = test_loss
            metrics["Test Accuracy"] = test_acc
        pprint(metrics)
        print("\n")

        wandb.log(metrics)

        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        if count > args.early_stop_patience:
            break
