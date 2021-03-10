## In the light of feature distributions: Moment matching for Neural Style Transfer (CVPR 2021)

This repository provides code to recreate results presented in [In the light of feature distributions: Moment matching for Neural Style Transfer](https://linktoarxiv.follows).

For more information, please see the [project website](https://linkfollowssoon.github.io)

<hr />
<img src="assets/teaser.jpg" />

### Contact
If you have any questions, please let me <a href="javascript:location='mailto:\u006e\u0069\u006b\u006f\u006c\u0061\u0069\u002e\u006b\u0061\u006c\u0069\u0073\u0063\u0068\u0065\u006b\u0040\u0067\u0065\u006f\u0064\u002e\u0062\u0061\u0075\u0067\u002e\u0065\u0074\u0068\u007a\u002e\u0063\u0068';void 0">know</a>

### Instructions
Running neural style transfer with Central Moment Discrepancy is as easy as running 
```shell
python main.py --c_img ./path/to/content.jpg --s_img ./path/to/style.jpg
``` 
You have the following command line arguments to change to your needs:
<pre>
  --c_img         The content image that is being stylized.
  --s_img         The style image
  --epsilon       Iterative optimization is stopped if delta value of 
                  moving average loss is smaller than this value.
  --max_iter      Maximum iterations if epsilon is not surpassed
  --alpha         Convex interpolation of style and content loss 
                  (should be set high > 0.9 since we start with content as target)
  --lr            Learning rate of Adam optimizer
  --im_size       Output image size. Can either be single integer for keeping aspect ratio or tuple.
</pre>

### Citations
```
Follows soon.
```
