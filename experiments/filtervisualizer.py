import matplotlib as plt

from fastai.conv_learner import *


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

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

    def visualize(self,
                  layer,
                  conv_filter,
                  lr=0.1,
                  opt_steps=20,
                  print_losses=False,
                  layer_name_plot=None,
                  blur=None, ):
        sz = self.size
        img = (np.random.random((sz, sz, 3)) * 20 + 128.) / 255.
        activations = SaveFeatures(layer)  # register hook

        for i in range(self.upscaling_steps
                       ):  # scale the image up upscaling_steps times

            if i > self.upscaling_steps / 2:
                opt_steps_ = int(opt_steps * 1.3)
            else:
                opt_steps_ = opt_steps
            train_tfms, val_tfms = tfms_from_model(self.model, sz)
            img_var = V(
                val_tfms(img)[None],
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

            img = val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(
                1, 2, 0))

            self.output = img
            sz = int(self.upscaling_factor * sz)  # calculate new image sz
            img = cv2.resize(img, (sz, sz),
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

    def get_transformed_img(self, img, sz):
        train_tfms, val_tfms = tfms_from_model(self.model, sz)

        if self.cpu:
            return val_tfms.denorm(np.rollaxis(val_tfms(img)[None], 1, 4))[0]

        return val_tfms.denorm(np.rollaxis(to_np(val_tfms(img)[None]), 1,
                                           4))[0]

    def most_activated(self, image, layer):
        train_tfms, val_tfms = tfms_from_model(self.model, 224)
        transformed = val_tfms(image)

        activations = SaveFeatures(layer)  # register hook
        self.model(V(transformed)[None])

        print(activations.features[0, 5].mean().data.cpu().numpy())
        mean_act = [
            activations.features[0, i].mean().data.cpu().numpy()
            for i in range(activations.features.shape[1])
        ]
        activations.close()

        return mean_act
