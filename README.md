## In the light of feature distributions: Moment matching for Neural Style Transfer (CVPR 2021)

This repository provides code to recreate results presented in [In the light of feature distributions: Moment matching for Neural Style Transfer](https://linktoarxiv.follows).

For more information, please see the [project website](https://linkfollowssoon.github.io)

<hr />
<img src="assets/teaser.jpg" />

### Contact
If you have any questions, please let me <a href="mailto:nikolai.kalischek@geod.baug.ethz.ch">know</a>

### Instructions
Running neural style transfer with Central Moment Discrepancy is as easy as running `main.py`. 
You have the following command line arguments to change to your needs:
<code>
  --c_img The content image that is being stylized.
  --s_img The style image
  --epsilon Iterative optimization is stopped if delta value of moving average loss is smaller than this value.
  --max_iter Maximum iterations if epsilon is not surpassed
  --alpha Convex interpolation of style and content loss (should be set high > 0.9 since we start with content as target)
  --lr Learning rate of Adam optimizer
  --im_size Output image size. Can either be single integer for keeping aspect ratio or tuple.
</code>

### Citations
