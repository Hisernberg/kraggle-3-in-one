# Video bout coherence: NEGATIVE result, no test changes proposed

The OOF video features were built exactly like the fork's (frames start*3//5+8 .. +22; rec index = sorted inertial_feat stems; start is in 50 Hz samples).
Within each subject: PCA-whitened mean of the 15 frames (top 0/2/3/5/10 PCs dropped, 32/64/96/128 components kept), then a kNN label share (k=5/10/20) and cosine similarity to the bout (class) centroids, using the current labels.

## Oracle separability (true labels as neighbours, OOF)
- 1-NN label agreement over the whole subject: 0.92-0.93 (fork accuracy 0.923).
- Within-group 10-NN vote: jog 0.87 | stretches 0.99 | push 0.95 | sit 0.97 | lunge 0.95.
- Only 34% of 1-NN pairs are within 2 s of each other, so video neighbours are not just adjacent seconds.

## Rule simulation (fork predictions as labels; F1 base 0.92304 / 0.91830)
| variant | best precision | dF1 |
|---|---|---|
| subject-wide kNN share (>=t, own<=1-t), any or confusable target | 0.12-0.36 | always < 0 |
| + target's fork probability >= q | <= 0.40 | <= -0.0002 |
| centroid margin (other minus own > t), confusable groups | 0.06-0.14 | < 0 |
| arm-only + target bout arm-deficit (arm fraction < 0.42) | 0.08-0.22 | < 0 |
| kNN inside each confusable group's predicted windows (best region t>=0.8, q>=0.1) | fork 0.48-0.60, fork2 0.29-0.57 (n = 7-80) | about +0.00005, sign flips with config |

No setting passes precision > 0.6 with dF1 > 0 on both fork and fork2. Fork's errors cluster in video space, so where the video vote disagrees with the label, the label is usually the one that is right.

## Sanity check, sbj_24 jog vs jog-butt (L11 labels, the 15 windows masked)
10-NN jog-butt share: verified group A [.4 .5 .3 .8 .7 .1 .5 .4 .4 .7 .7] vs wrong group B [.3 .4 .5 .3].
Centroid margins overlap the same way. The A mean is higher (0.50 vs 0.38) but the groups are not separable: window 1844 (A) is at 0.1 and window 9927 (B) at 0.5.

## Test changes
None. Confidence that video bout coherence gives no usable repair on top of fork/L11: high.
