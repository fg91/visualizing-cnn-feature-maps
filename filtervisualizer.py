import matplotlib as plt

from fastai.conv_learner import *


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, _input, output):
        self.features = output

    def close(self):
        self.hook.remove()


class FilterVisualizer():
    def __init__(self,
                 model,
                 size=56,
                 upscaling_steps=12,
                 upscaling_factor=1.2,
                 cpu=False):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = model
        set_trainable(self.model, False)
        self.cpu = cpu
        self.train_tfms, self.val_tfms = tfms_from_model(self.model, self.size)
        self.output = None

    def visualize(self,
                  layer,
                  conv_filter,
                  lr=0.1,
                  opt_steps=20,
                  print_losses=False,
                  layer_name_plot=None,
                  blur=None, ):
        img = (np.random.random((self.size, self.size, 3)) * 20 + 128.) / 255.
        activations = SaveFeatures(layer)  # register hook

        for i in range(self.upscaling_steps
                       ):  # scale the image up upscaling_steps times

            if i > self.upscaling_steps / 2:
                opt_steps_ = int(opt_steps * 1.3)
            else:
                opt_steps_ = opt_steps
            img_var = V(
                self.val_tfms(img)[None],
                requires_grad=True)  # convert image to variable that requires grad
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

            for n in range(
                    opt_steps_):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, conv_filter].mean()

                if print_losses and i % 3 == 0 and n % 5 == 0:
                    print(f'{i} - {n} - {float(loss)}')
                loss.backward()
                optimizer.step()

            img = self.val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(
                1, 2, 0))

            self.output = img
            up_size = int(self.upscaling_factor * self.size)  # calculate new image size
            img = cv2.resize(img, (up_size, up_size),
                             interpolation=cv2.INTER_CUBIC)  # scale image up

            if blur:
                img = cv2.blur(
                    img,
                    (blur,
                     blur))  # blur image to reduce high frequency patterns

        if layer_name_plot:
            self.save(layer_name_plot, conv_filter)
        activations.close()

        return np.clip(self.output, 0, 1)

    def save(self, layer_name_plot, conv_filter):
        plt.imsave(f'layer_{layer_name_plot}_conv_filter_{conv_filter}.png',
                   np.clip(self.output, 0, 1))

    def get_transformed_img(self, img):
        if self.cpu:
            return self.val_tfms.denorm(np.rollaxis(self.val_tfms(img)[None], 1, 4))[0]

        return self.val_tfms.denorm(np.rollaxis(to_np(self.val_tfms(img)[None]), 1,
                                                4))[0]

    def most_activated(self, image, layer):
        transformed = self.val_tfms(image)

        activations = SaveFeatures(layer)  # register hook
        self.model(V(transformed)[None])

        print(activations.features[0, 5].mean().data.cpu().numpy())
        mean_act = [
            activations.features[0, i].mean().data.cpu().numpy()
            for i in range(activations.features.shape[1])
        ]
        activations.close()

        return mean_act
