# 0x05. Regularization

## Resources

- [Regularization (mathematics)](https://intranet.hbtn.io/rltoken/G22TZHYwwb0PwlAuEZdDEQ)
- [An Overview of Regularization Techniques in Deep Learning](https://intranet.hbtn.io/rltoken/Mao_NUBBiwm0Qh8b-axAgw)
- [L2 Regularization and Back-Propagation](https://intranet.hbtn.io/rltoken/AY80ruaSMDL_AGnjZOpWGQ)
- [Intuitions on L1 and L2 Regularisation](https://intranet.hbtn.io/rltoken/OUTExT9leQf9sz5Dg0sF0Q)
- [Analysis of Dropout](https://intranet.hbtn.io/rltoken/huRNIkxWr5OV1Tit658LcQ)
- [Early stopping](https://intranet.hbtn.io/rltoken/4YMCmw41ovvYtMvr-Wl7LA)
- [How to use early stopping properly for training deep neural network?](https://intranet.hbtn.io/rltoken/t6UPkGJXD_nK7TfGwE9Rig)
- [Data Augmentation | How to use Deep Learning when you have Limited Data ](https://intranet.hbtn.io/rltoken/MaLMSTSCPux71mW1RIhiBA)
- [deeplearning.ai](https://intranet.hbtn.io/rltoken/GriJE79Gr4BF8HG2DGpbYg)
- [Regularization](https://intranet.hbtn.io/rltoken/BJoxOnJN-GJyZ_fJ9qT0EQ)
- [Why Regularization Reduces Overfitting](https://intranet.hbtn.io/rltoken/dLdv5Gi77DmWNyR3MHe69g)
- [Dropout Regularization](https://intranet.hbtn.io/rltoken/23ue4EQxNd9LOCW0Q6FNNQ)
- [Understanding Dropout](https://intranet.hbtn.io/rltoken/eleB8ZvoJiOltULeHkDvGQ)
- [Other Regularization Methods](https://intranet.hbtn.io/rltoken/QuFgq0_MKTGq6UAKj5OjEw)

References :

- [numpy.linalg.norm](https://intranet.hbtn.io/rltoken/5YoCQBn6-nRyuldXYANpuw)
- [numpy.random.binomial](https://intranet.hbtn.io/rltoken/vdPHIWg_6Dq6-e6Wvjmz9w)
- [tf.keras.regularizers.L2](https://intranet.hbtn.io/rltoken/y9OSDn67_DpM5hlMXe116Q)
- [tf.layers.Dense](https://intranet.hbtn.io/rltoken/K0y9uk5aa5uzLsyavooezg)
- [tf.losses.get_regularization_loss](https://intranet.hbtn.io/rltoken/R0pALpDYtCoQulGJDTE52A)
- [tf.layers.Dropout](https://intranet.hbtn.io/rltoken/VzdLxZHGgNTASxpaeyymDA)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://intranet.hbtn.io/rltoken/2jIHjQpd_A2-4IF1SbL5dg)
- [Early Stopping - but when?](https://intranet.hbtn.io/rltoken/b_knZ8MqBEHA3TPoGruYGw)
- [L2 Regularization versus Batch and Weight Normalization](https://intranet.hbtn.io/rltoken/JVvKoC0p-wBoLl3qF7xChQ)

## Learning Objectives

- What is regularization? What is its purpose?
- What is are L1 and L2 regularization? What is the difference between the two methods?
- What is dropout?
- What is early stopping?
- What is data augmentation?
- How do you implement the above regularization methods in Numpy? Tensorflow?
- What are the pros and cons of the above regularization methods?

## Tasks

| Filename                      | Description                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 0-l2_reg_cost.py              | Function that calculates the cost of a neural network with L2 regularization                                   |
| 1-l2_reg_gradient_descent.py  | Function that updates the weights and biases of a neural network using gradient descent with L2 regularization |
| 2-l2_reg_cost.py              | Function that calculates the cost of a neural network with L2 regularization                                   |
| 3-l2_reg_create_layer.py      | Function that creates a tensorflow layer that includes L2 regularization                                       |
| 4-dropout_forward_prop.py     | Function that conducts forward propagation using Dropout                                                       |
| 5-dropout_gradient_descent.py | Function that updates the weights of a neural network with Dropout regularization using gradient descent       |
| 6-dropout_create_layer.py     | Function that creates a layer of a neural network using dropout                                                |
| 7-early_stopping.py           | Function that determines if you should stop gradient descent early                                             |
