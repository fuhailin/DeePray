# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import, division, print_function

import re

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


class AdamWeightDecayOptimizer(tf.keras.optimizers.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):

    def apply_gradients1(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

class Lookahead(optimizer.Optimizer):
    """Tensorflow implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    Reference: https://github.com/michaelrzhang/lookahead/blob/master/lookahead_tensorflow.py
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, use_locking=False, name="Lookahead"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        """
        super(Lookahead, self).__init__(use_locking, name)
        self.optimizer = optimizer
        self._la_step = 0
        self._la_alpha = la_alpha
        self._total_la_steps = la_steps

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

        self._var_list = var_list
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self._la_step,
                                       name="la_step",
                                       colocate_with=first_var)

        # Create slots for the cached parameters.
        for v in var_list:
            self._zeros_slot(v, "cached_params", self._name)

    def _prepare(self):
        self.optimizer._prepare()

        la_alpha = self._call_if_callable(self._la_alpha)
        total_la_steps = self._call_if_callable(self._total_la_steps)

        self._la_alpha_t = ops.convert_to_tensor(la_alpha, name="la_alpha")
        self._total_la_steps_t = ops.convert_to_tensor(total_la_steps, name="total_la_steps")

    def _get_la_step_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return self._get_non_slot_variable("la_step", graph=graph)

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        inner_finish_op = self.optimizer._finish(update_ops, name_scope)

        with ops.control_dependencies([inner_finish_op, ]):
            la_step = self._get_la_step_accumulators()
            with ops.colocate_with(la_step):
                def update_la_step_func():
                    # update the la_step
                    return control_flow_ops.group([la_step.assign(
                        la_step + 1, use_locking=self._use_locking), ])

                def pull_back_func():
                    # update the la_step
                    update_la_step = la_step.assign(
                        0, use_locking=self._use_locking)
                    # interpolate the variables
                    interpolation = [v.assign(
                        self.get_slot(v, "cached_params") + self._la_alpha_t * (v - self.get_slot(v, "cached_params")))
                                     for v in self._var_list]

                    # update the cached params
                    with ops.control_dependencies(interpolation):
                        update_cached_params = [self.get_slot(v, "cached_params").assign(updated_v) for v, updated_v in
                                                zip(self._var_list, interpolation)]
                    return control_flow_ops.group([update_la_step, ] + interpolation + update_cached_params)

                # condition for when to pull back the params
                condition = tf.greater_equal(la_step, self._total_la_steps_t)
                update_lookahead_states = tf.cond(condition,
                                                  pull_back_func,
                                                  update_la_step_func,
                                                  )

        return control_flow_ops.group([inner_finish_op, update_lookahead_states],
                                      name=name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param

@tf_export("train.RAdamOptimizer")
class RAdamOptimizer(optimizer.Optimizer):
  """Optimizer that implements the RAdam algorithm.
  See [Liyuan Liu et al., 2019](https://arxiv.org/abs/1908.03265)
  ([pdf](https://arxiv.org/pdf/1908.03265.pdf)).
  Reference: https://github.com/lifeiteng/Optimizers/blob/master/radam_optimizer.py
  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="RAdam"):
    """Construct a new Rectified Adam optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "RAdam".
    @compatibility(eager)
    When eager execution is enabled, `learning_rate`, `beta1`, `beta2`, and
    `epsilon` can each be a callable that takes no arguments and returns the
    actual value to use. This can be useful for changing these values across
    different invocations of optimizer functions.
    @end_compatibility
    """
    super(RAdamOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Created in SparseApply if needed.
    self._updated_lr = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("step", graph=graph),
              self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=1.0,
                                   name="step",
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=self._beta1,
                                   name="beta1_power",
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=self._beta2,
                                   name="beta2_power",
                                   colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense_shared(self, grad, var, m, v):
    step, beta1_power, beta2_power = self._get_beta_accumulators()
    step = math_ops.cast(step, var.dtype.base_dtype)
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m_t = state_ops.assign(m, m * beta1_t + grad * (1.0 - beta1_t),
                           use_locking=self._use_locking)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v_t = state_ops.assign(v, v * beta2_t + (grad * grad) * (1.0 - beta2_t),
                           use_locking=self._use_locking)

    rho_inf = math_ops.cast(2.0 / (1.0 - self._beta2) - 1.0, var.dtype.base_dtype)
    rho_t = rho_inf - step * (2.0 * beta2_power / (1.0 - beta2_power))

    r_t = math_ops.sqrt(
        (1.0 - beta2_power) * ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))

    update = control_flow_ops.cond(math_ops.greater(rho_t, 5.0),
                                   true_fn=lambda: (lr_t / (1.0 - beta1_power) * r_t) * (
                                       m_t / (math_ops.sqrt(v_t) + epsilon_t)),
                                   false_fn=lambda: (lr_t / (1.0 - beta1_power)) * m_t)

    var_update = state_ops.assign_sub(var, update,
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return self._apply_dense_shared(grad, var, m, v)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    step, beta1_power, beta2_power = self._get_beta_accumulators()
    step = math_ops.cast(step, var.dtype.base_dtype)
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)

    rho_inf = math_ops.cast(2.0 / (1.0 - self._beta2) - 1.0, var.dtype.base_dtype)
    rho_t = rho_inf - step * (2.0 * beta2_power / (1.0 - beta2_power))

    r_t = math_ops.sqrt(
        (1.0 - beta2_power) * ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))

    update = control_flow_ops.cond(math_ops.greater(rho_t, 5.0),
                                   true_fn=lambda: (lr_t / (1.0 - beta1_power) * r_t) * (
                                       m_t / (math_ops.sqrt(v_t) + epsilon_t)),
                                   false_fn=lambda: (lr_t / (1.0 - beta1_power)) * m_t)

    var_update = state_ops.assign_sub(var, update,
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x, i, v, use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(
            x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(
        grad, var, indices, self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      step, beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_step = step.assign(
            step + 1.0, use_locking=self._use_locking)
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_step, update_beta1, update_beta2],
                                  name=name_scope)


class AMSGrad(optimizer.Optimizer):
    """Optimizer that implements the AMSGrad algorithm.
    See (https://openreview.net/pdf?id=ryQu7f-RZ)

    """

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, use_locking=False, name="AMSGrad"):
        """Construct a new AMSGrad optimizer.
        Args:
            learning_rate: A Tensor or a floating point value. The
                learning rate.
            beta1: A float value or a constant float tensor. The
                exponential decay rate for the 1st moment estimates.
            beta2: A float value or a constant float tensor. The
                exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            name: Optional name for the operations created when applying gradients.
            Defaults to "AMSGrad".
        """
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return
        raise NotImplementedError("Sparse gradient updates are not supported yet.")