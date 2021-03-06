import sys
import argparse
import matplotlib.pyplot as plt
import timeit

import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch.nn as nn
from torch.optim import Adam

from dataset import SingleStyleData
from model import VGG
from losses import ContentLoss, StyleLoss


class CMDStyleTransfer:

    def __init__(self, content_img, style_img, image_size=(512, 512), lr=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.dataset = self._init_dataset(content_img, style_img)
        self.input_img = self.dataset[0][0].clone().to(self.device)
        self.model = self._init_model()
        self.lr = lr
        self.optim = Adam([self.input_img.requires_grad_()], lr=self.lr)

    def _init_dataset(self, content_img, style_img):
        transform = transforms.Compose([transforms.Resize(self.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)])

        return SingleStyleData(path_content=content_img,
                               path_style=style_img,
                               device=self.device,
                               transform=transform)

    def _init_model(self):
        vgg = vgg19(pretrained=False).eval()
        vgg.load_state_dict(torch.load('./weights/vgg19-dcbb9e9d.pth'), strict=False)
        # print(list(vgg.children()))
        modules = list(vgg.children())[0][:29]
        # Replace inplace ReLU as it
        for i, layer in enumerate(modules):
            if isinstance(layer, nn.ReLU):
                modules[i] = nn.ReLU(inplace=False)

        vgg_style = VGG(modules).cuda()
        for p in vgg_style.parameters():
            p.requires_grad = False
        return vgg_style

    def run(self, vgg_weights, alpha=0.9, style_loss_weights=(1, 1, 1, 1, 1), epsilon=0.0001, max_iter=500):
        # Compute features of target once
        img_c, img_s = self.model(self.dataset[0][0]), self.model(self.dataset[0][1])
        conv_c = torch.nn.ReLU()(img_c[3]).detach()
        convs_s = [torch.sigmoid(s).detach() for s in img_s]

        # Initialize losses
        c_loss = ContentLoss().cuda()
        s_losses = [StyleLoss(conv_s, k=len(style_loss_weights), weights=style_loss_weights).cuda() for conv_s in convs_s]

        # Start iterative optimization
        moving_loss = None
        iteration = 0
        # Measure computation time
        starttime = timeit.default_timer()
        while iteration < max_iter:
            # Feed current image into model
            outputs = self.model(self.input_img)
            loss_c = c_loss(torch.nn.ReLU()(outputs[3]), conv_c)
            loss_s = torch.tensor(0.0).cuda()
            for i, (conv_o, conv_s) in enumerate(zip(outputs, convs_s)):
                loss_s += vgg_weights[i] * s_losses[i](torch.sigmoid(conv_o))
            loss = (1 - alpha) * loss_c + alpha * loss_s
            # Optimize
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            new_loss = loss_s.item()
            iteration += 1
            if moving_loss is None:
                moving_loss = loss_s.item()
                continue
            else:
                moving_loss = 0.99*moving_loss + 0.01*new_loss
            if iteration % 50 == 0:
                print("Current iteration is {} and the loss is {}".format(iteration, moving_loss))
            if abs(moving_loss - new_loss) <= epsilon:
                print("Delta smaller than eps.")
                print("Current iteration is {} and the loss is {}".format(iteration, moving_loss))
                break
        difference = timeit.default_timer() - starttime
        print("The time difference is :", difference)

    def get_current_image(self):
        image = self.input_img.clone().detach().cpu()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = image * torch.tensor(self.std).view(-1, 1, 1) + torch.tensor(self.mean).view(-1, 1, 1)
        image.data.clamp_(0, 1)
        return transforms.ToPILImage()(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0.001, help='Delta difference stopping criterion')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iteration if delta not reached')
    parser.add_argument('--alpha', type=float, default=1.0, help='Style and content loss weighting.'
                                                                 'Needs to be high since we are starting with content'
                                                                 'image.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate used during optimization')
    parser.add_argument('--c_img', type=str, default='./images/content/pablo_picasso.jpg', help='Path to content image')
    parser.add_argument('--s_img', type=str, default='./images/style/picasso.jpg', help='Path to style image')
    parser.add_argument('--im_size', type=int, default=512, nargs='+', help='Image size. Either single int or tuple of int')
    args = parser.parse_args()

    vgg_weights = [1e3/n**2 for n in [64, 128, 256, 512, 512]]
    img_size = args.im_size  # Integer or list (512, 512)
    if isinstance(img_size, list):
        if len(args.im_size) > 2:
            print("Image size can either be a single int or a list of two ints.")
            sys.exit(0)

    # Init CMD style transfer
    cmd_transfer = CMDStyleTransfer(args.c_img, args.s_img, img_size, lr=args.lr)
    cmd_transfer.run(vgg_weights=vgg_weights, alpha=args.alpha, epsilon=args.epsilon, max_iter=args.max_iter)
    stylized_image = cmd_transfer.get_current_image()

    # Plot result
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    plt.axis('off')
    plt.title('Stylized image')
    plt.imshow(stylized_image)
    plt.show()
