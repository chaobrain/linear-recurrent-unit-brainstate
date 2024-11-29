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


import brainstate as bst
import jax
import jax.numpy as jnp

__all__ = [
    'LRULayer',
    'LRUBlock',
    'EncoderLayer',
    'OfflineModel'
]


@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


class LRULayer(bst.nn.Module):
    r"""
    `Linear Recurrent Unit <https://arxiv.org/abs/2303.06349>`_ (LRU) layer.

    This layer processes a sequence of inputs. At each time step, it updates the hidden state

    .. math::

       h_{t+1} = \lambda * h_t + \exp(\gamma^{\mathrm{log}}) B x_{t+1} \\
       \lambda = \text{diag}(\exp(-\exp(\nu^{\mathrm{log}}) + i \exp(\theta^\mathrm{log}))) \\
       y_t = Re[C h_t + D x_t]

    Args:
        d_hidden: int
            Hidden state dimension.
        d_model: int
            Input and output dimensions.
        r_min: float, optional
            Smallest lambda norm.
        r_max: float, optional
            Largest lambda norm.
        max_phase: float, optional
            Max phase lambda.
    """

    def __init__(
        self,
        d_hidden: int,  # hidden state dimension
        d_model: int,  # input and output dimensions
        r_min: float = 0.0,  # smallest lambda norm
        r_max: float = 1.0,  # largest lambda norm
        max_phase: float = 6.28,  # max phase lambda
    ):
        super().__init__()

        self.in_size = d_model
        self.out_size = d_model

        self.d_hidden = d_hidden
        self.d_model = d_model
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        # theta parameter
        theta_log = jnp.log(max_phase * bst.random.uniform(size=d_hidden))
        self.theta_log = bst.ParamState(theta_log)

        # nu parameter
        u = bst.random.uniform(size=d_hidden)
        nu_log = jnp.log(-0.5 * jnp.log(u * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        self.nu_log = bst.ParamState(nu_log)

        # gamma parameter
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))
        self.gamma_log = bst.ParamState(gamma_log)

        # Glorot initialized Input/Output projection matrices
        B_re = bst.random.randn(d_model, d_hidden) / jnp.sqrt(2 * d_model)
        self.B_re = bst.ParamState(B_re)

        B_im = bst.random.randn(d_model, d_hidden) / jnp.sqrt(2 * d_model)
        self.B_im = bst.ParamState(B_im)

        C_re = bst.random.randn(d_hidden, d_model) / jnp.sqrt(d_hidden)
        self.C_re = bst.ParamState(C_re)

        C_im = bst.random.randn(d_hidden, d_model) / jnp.sqrt(d_hidden)
        self.C_im = bst.ParamState(C_im)

        # Parameter for skip connection
        D = bst.random.randn(d_model)
        self.D = bst.ParamState(D)

    def update(self, inputs):
        # inputs: [n_seq, d_model]

        # input processing
        B_norm = (self.B_re.value + 1j * self.B_im.value) * jnp.exp(self.gamma_log.value)
        Bu_elements = jax.vmap(lambda u: u @ B_norm)(inputs)  # [n_seq, d_hidden]

        # recurrent processing
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log.value) + 1j * jnp.exp(self.theta_log.value))
        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)  # [n_seq, d_hidden]
        _, hidden_states = jax.lax.associative_scan(binary_operator_diag, (Lambda_elements, Bu_elements))

        # output processing
        C = self.C_re.value + 1j * self.C_im.value
        y = jax.vmap(lambda h, x: jnp.real(jnp.dot(h, C)) + self.D.value * x)(hidden_states, inputs)
        return y  # [n_seq, d_model]


class LRUBlock(bst.nn.Module):
    r"""
    Single layer, with LRU module, GLU, dropout and batch/layer norm.

    The computation is as follows:

    .. math::

         \text{x} = \text{dropout}(\text{GLU}(\text{LRU}(\text{norm}(\text{inputs})))) \\
         \text{out} = \text{inputs} + \text{dropout}(\text{Linear}(\text{x}) \odot \text{sigmoid}(\text{Linear}(\text{x})))

    Args:
        d_hidden: int
            Hidden state dimension.
        d_model: int
            Input and output dimensions.
        r_min: float, optional
            Smallest lambda norm.
        r_max: float, optional
            Largest lambda norm.
        dropout: float, optional
            Dropout probability.
        norm: str, optional
            Type of normalization. It can be 'batchnorm', 'layernorm', 'rmsnorm' or 'none'.

            - 'batchnorm': Batch normalization.
            - 'layernorm': Layer normalization.
            - 'rmsnorm': Root mean square normalization.
            - 'none': No normalization.
        max_phase: float, optional
            Max phase lambda.
    """

    def __init__(
        self,
        d_hidden: int,
        d_model: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        dropout: float = 1.0,  # dropout probability
        norm: str = 'batch',  # type of normalization
        max_phase: float = 6.28
    ):
        super().__init__()

        self.in_size = d_model
        self.out_size = d_model
        self.drop = bst.nn.Dropout1d(prob=1 - dropout, channel_axis=-1)  # dropout mask shared across time
        self.lru = LRULayer(
            d_hidden=d_hidden,
            d_model=d_model,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase
        )
        if norm == 'batchnorm':
            self.norm = bst.nn.BatchNorm1d([None, d_model], axis_name='batch')
        elif norm == 'layernorm':
            self.norm = bst.nn.LayerNorm([d_model])
        elif norm == 'rmsnorm':
            self.norm = bst.nn.RMSNorm([d_model])
        elif norm == 'none':
            self.norm = lambda x: x
        else:
            raise ValueError(f"Normalization {norm} not recognized")
        self.out1 = bst.nn.Linear(d_model, d_model, w_init=bst.init.LecunNormal())
        self.out2 = bst.nn.Linear(d_model, d_model, w_init=bst.init.LecunNormal())

    def update(self, inputs):
        self.norm.in_size = inputs.shape  # avoid shape mismatch when using batchnorm
        x = self.norm(inputs)  # pre-layer normalization
        x = self.lru(x)  # LRU
        x = self.drop(bst.functional.gelu(x))  # post layer activation and dropout
        x = self.out1(x) * bst.functional.sigmoid(self.out2(x))  # output layer
        x = self.drop(x)
        return inputs + x  # skip connection


class EncoderLayer(bst.nn.Module):
    r"""
    Encoder containing stacked LRU blocks.

    Args:
        n_layers: int
            Number of layers in the network.
        d_hidden: int
            Hidden state dimension.
        d_model: int
            Input and output dimensions.
        r_min: float, optional
            Smallest lambda norm.
        r_max: float, optional
            Largest lambda norm.
        dropout: float, optional
            Dropout probability.
        norm: str, optional
            Type of normalization. It can be 'batchnorm', 'layernorm', 'rmsnorm' or 'none'.

            - 'batchnorm': Batch normalization.
            - 'layernorm': Layer normalization.
            - 'rmsnorm': Root mean square normalization.
            - 'none': No normalization.
        max_phase: float, optional
            Max phase lambda.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_model: int,
        n_layers: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        dropout: float = 0.0,
        norm: str = 'rmsnorm',
        max_phase: float = 6.28
    ):
        super().__init__()

        self.encoder = bst.nn.Linear(d_input, d_model, w_init=bst.init.LecunNormal())
        blocks = [
            LRUBlock(
                d_hidden=d_hidden,
                d_model=d_model,
                r_min=r_min,
                r_max=r_max,
                dropout=dropout,
                norm=norm,
                max_phase=max_phase
            )
            for _ in range(n_layers)
        ]
        self.blocks = bst.nn.Sequential(*blocks)

    def update(self, x):
        x = self.encoder(x)  # embed input in latent space
        x = self.blocks(x)  # apply each layer
        return x


class OfflineModel(bst.nn.Module):
    """
    Stacked encoder and decoder with pooling and softmax.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_model: int,
        d_output: int,
        n_layers: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        dropout: float = 0.0,
        norm: str = 'rmsnorm',
        pooling: str = 'mean',
        max_phase: float = 6.28,
        multidim: int = 1  # number of outputs
    ):
        super().__init__()

        # parameters
        self.pooling = pooling
        self.multidim = multidim
        self.d_output = d_output

        # encoder
        self.encoder = EncoderLayer(
            d_input=d_input,
            d_hidden=d_hidden,
            d_model=d_model,
            n_layers=n_layers,
            r_min=r_min,
            r_max=r_max,
            dropout=dropout,
            norm=norm,
            max_phase=max_phase
        )

        # decoder
        self.decoder = bst.nn.Linear(d_model, d_output * multidim, w_init=bst.init.LecunNormal())

    def update(self, xs):
        # xs: [n_seq, ...]

        # encoding
        xs = self.encoder(xs)

        # pooling
        if self.pooling == "mean":
            xs = jnp.mean(xs, axis=0)
        elif self.pooling == "last":
            xs = xs[-1]
        elif self.pooling == "max":
            xs = jnp.max(xs, axis=0)
        elif self.pooling == "min":
            xs = jnp.min(xs, axis=0)
        elif self.pooling == "sum":
            xs = jnp.sum(xs, axis=0)
        elif self.pooling == "none":
            pass
        else:
            raise ValueError(f"Pooling mode {self.pooling} not recognized")

        # decoding
        xs = self.decoder(xs)
        if self.multidim > 1:
            xs = jnp.reshape(xs, xs.shape[:-1] + (self.d_output, self.multidim))

        return bst.functional.log_softmax(xs, axis=-1)
