# Proposed outer product in Tensorflow
Ref https://github.com/tensorflow/tensorflow/issues/17564


Tensor (outer product) is the fundamental operation on tensors, but there appears to be no method tf.outer in tensorflow analogous to np.outer in numpy to compute the outer product of arbitrary tensors. A google search pulls up these implementation suggestions on stackoverflow: https://stackoverflow.com/questions/33858021/outer-product-in-tensorflow, but these require the dimensions of the tensors to be accessed/known beforehand.

It would be nice to express this in terms of a pairwise (given associativity) operation tf.outer.

The tensor product for an arbitrary collection of tensors can be computed:
```
def tensor_product(*e):
    """ Tensor product of elements """
    if len(e) == 1:
        return e
    elif len(e) == 2:
        a, b = e
        r_a = len(a.get_shape().as_list())
        r_b = len(b.get_shape().as_list())
        s_a = tf.concat([tf.shape(a), tf.constant([1] * r_b)], axis=0)
        s_b = tf.concat([tf.constant([1] * r_a), tf.shape(b)], axis=0)
        a_reshaped = tf.reshape(a, s_a)
        b_reshaped = tf.reshape(b, s_b)
        return a_reshaped * b_reshaped
    prod = e[0]
    for tensor in e[1:]:
        prod = tensor_product(prod, tensor)
    return prod
```

The tensor product allows more elegant expression of rank-2 and greater tensors in a loss function.
For example here is a diagonal and elliptical (weighted terms) quadratic term:
```
def diagonal_M(batch_size, d):
    """ M_abij = delta_ab delta_ij """
    return tensor_product(tf.diag([1.] * batch_size), tf.diag([1.] * d))
```
```
def biased_diagonal_M(batch_size, d):
    """ M_abij = lambda_i delta_ab delta_ij """
    return tensor_product(tf.diag([1.] * batch_size), tf.diag([5.] + [1.] * (d-1)))
```    

