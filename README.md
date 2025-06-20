<h2>Supervised Classification</h2>

<img src="./results/cnn/validation_metrics_CNN.png"
style="width:7.1875in;height:2.78125in" /><img src="./g5k5uwcj.png"
style="width:6.27083in;height:3.76042in" />

VAE

q1:Loss Curves

The total loss and MSE decrease significantly in the first 10 epochs and
then plateau, indicating successful training.

The KL divergence increases steadily and plateaus, as wanted with better
latent regularization.

The gap between training and validation loss is small, pointing to no
harsh overfitting.

<img src="./5ueenm3o.png"
style="width:6.27083in;height:3.4375in" /><img src="./0k5lc0pb.png"
style="width:6.27083in;height:3.45833in" />

Training:

Validation:

Epoch 1: Reconstructions are blurry, barely resembling digits.

Epoch 5-10: Shapes become sharper, and digit identity is typically
preserved. Epoch 20-30: Reconstructions continue to get better, with
strong fidelity to input.

The small discrepancy in MSE and visually similar quality of
reconstruction for training and validation samples shows that the model
did not overfit the training data significantly.

We can also see that in both of the given 10 digits the model gets 1
wrong and at different values

<img src="./iuqcspxi.png"
style="width:6.27083in;height:1.875in" /><img src="./rlqqv3bv.png"
style="width:6.27083in;height:3.19792in" />

Q2:

q3 Training:

<img src="./afq3ln3x.png"
style="width:6.125in;height:3.21875in" />Validation:

<img src="./dklt13x1.png"
style="width:6.27083in;height:1.875in" /><img src="./adln5ej1.png" style="width:6.27083in;height:1.25in" />

The latent optimization VAE produces reconstructions that often fail to
preserve the digit's class. The digits are frequently distorted and
unrecognizable. This indicates that it can lead to representations that
sacrifice semantic meaning and class information.

So we can conclude that the amortized VAE produces better *q* vectors
mainly in terms of preserving the class identity of the digits.

Samples:

The samples from the latent optimization VAE are poor but appear like
recognizable digits. They appear like blurry digits resembling some time
to more than one, in compression the samples from the amortized VAE are
recognizable digits resembling only one digit, with some light
blurriness.

The initialization was sufficient, we can see that in some cases we
start the same and have the same latent vectors. We in both cases get
reasonable latent vectors and in the amortized we even get good
predictions.

Q4 (a)<u>Amortized</u>

> <img src="./ip4l4l03.png" style="width:6.27083in;height:1.25in" /><u>Latent
> Optimization</u>
>
> <u>Amortized</u> <u>Latent Optimization</u>

\(b\) Average log-probabilities per digit

> 0 : -217.58871459960938 1 : -68.08492279052734 2 : -295.7515869140625
> 3 : -238.4097900390625 4 : -206.52548217773438 5 : -301.0711975097656
> 6 : -231.7938690185547 7 : -144.23634338378906 8 : -295.1378479003906
>
> 9 : -161.9153289794922

-514.7296142578125 -137.36550903320312 -484.9649963378906
-428.3619079589844 -330.36676025390625 -484.19464111328125
-463.27215576171875 -324.5142517089844 -478.99755859375

-375.8153991699219

Digit 1 has the highest (least negative) log-probability in both
cases.Digit 1 has a very consistent and simple structure (a vertical
line), which the VAE can model well, leading to higher
log-probability.More complex digits (like 2, 5, 8) show greater
variability and are harder to model, resulting in lower

log-probabilities.

\(c\) Average log-probability (train)

> -208.86781311035156 -414.5937194824219 Average log-probability (test)
>
> -224.47113037109375 -384.385009765625

||
||
||

<u>LLM USE</u>

LLMs assisted in this exercise in the following ways:

> ● Code: LLMs helped write and refactor code.
>
> ● Plots: LLMs generated code for visualizations like loss curves and
> reconstructions.

LLMs were particularly useful for:

> ● Refactoring code for multiple inputs.
