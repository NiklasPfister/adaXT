# Honest splitting

Decision trees generally use the training data twice during fitting: Once for
deciding where to create splits resulting in the tree structure and once to
create a prediction for each leaf node.

While such a double use of the data may lead to overfitting, the effect is often
negligible, in particular when otherwise regularizing (e.g., fitting a forest or
constraining the maximum depth). Nevertheless it can be beneficial to adapt the
splitting to improve generalization performance. One way of achieving this is
via _honest splitting_
([Athey and Imbens, 2016](https://doi.org/10.1073/pnas.1510489113)). The
approach was originally developed in the context of causal effect estimation,
where the bias is shown to be reduced by honest splitting allowing for inference
on the causal effects.

Below we provide a short overview over honest splitting and explain how to use
it in adaXT.

## Two types of honest splitting

## Honest splitting in adaXT
