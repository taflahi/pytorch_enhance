import dataloader
import numpy as np
import network
import time
import torch
import torch.optim as optim
import sys

dataloader.args['train_scales'] = 2
dataloader.args['batch_size'] = 15
dataloader.args['train'] = '../images/*.jpg'

loader = dataloader.DataLoader()

# pretrain
pretrain_params = {
    'smoothness-weight': 1e7,
    'adversary-weight': 0.0,
    'generator-start': 0,
    'discriminator-start': 1,  # 1
    'adversarial-start': 2,  # 2
    'perceptual-weight': 1e0,
    'epochs': 50,  # 50
    'epoch-size': 72,  # 72
    'batch-size': 15,  # 15
    'image-size': 192,
    'zoom': 2,
    'learning-rate': 1e-4,
    'discriminator-size': 32
}

# pretrain
train_params = {
    'smoothness-weight': 2e4,
    'adversary-weight': 1e3,
    'generator-start': 5,
    'discriminator-start': 0,
    'adversarial-start': 5,
    'perceptual-weight': 1e0,
    'epochs': 250,
    'epoch-size': 72,
    'batch-size': 15,
    'image-size': 192,
    'zoom': 2,
    'learning-rate': 1e-4,
    'discriminator-size': 64
}


def train(enhancer, mode, param):
        # create initial discriminator
    disc = network.Discriminator(param['discriminator-size'])
    assert disc.channels == enhancer.discriminator.channels

    seed_size = param['image-size'] // param['zoom']
    images = np.zeros(
        (param['batch-size'], 3, param['image-size'], param['image-size']), dtype=np.float32)
    seeds = np.zeros((param['batch-size'], 3, seed_size,
                      seed_size), dtype=np.float32)

    loader.copy(images, seeds)
    # initial lr
    lr = network.decay_learning_rate(param['learning-rate'], 75, 0.5)

    # optimizer for generator
    opt_gen = optim.Adam(enhancer.generator.parameters(), lr=0)
    opt_disc = optim.Adam(disc.parameters(), lr=0)

    try:
        average, start = None, time.time()
        for epoch in range(param['epochs']):
            adversary_weight = 5e2

            total, stats = None, None

            l_r = next(lr)
            network.update_optimizer_lr(opt_gen, l_r)
            network.update_optimizer_lr(opt_disc, l_r)

            for step in range(param['epoch-size']):
                enhancer.zero_grad()
                disc.zero_grad()

                loader.copy(images, seeds)

                # run full network once
                gen_out, c12, c22, c32, c52, disc_out = enhancer(images, seeds)

                # clone discriminator on the full network
                enhancer.clone_discriminator_to(disc)

                # output of new cloned network (maybe you can assert it to
                # equal disc_out)
                disc_out2 = disc(c12.detach(), c22.detach(), c32.detach())
                disc_out_mean = disc_out2.sum(1).sum(1).sum(1).data.numpy()
                stats = stats + disc_out_mean if stats is not None else disc_out_mean

                # compute generator loss
                if mode == 'pretrain':
                    gen_loss = network.loss_perceptual(c22[:param['batch-size']], c22[param['batch-size']:]) * param['perceptual-weight'] \
                        + network.loss_total_variation(gen_out) * param['smoothness-weight'] \
                        + network.loss_adversarial(disc_out[1:]) * adversary_weight
                else:
                    gen_loss = network.loss_perceptual(c52[:param['batch-size']], c52[param['batch-size']:]) * param['perceptual-weight'] \
                        + network.loss_total_variation(gen_out) * param['smoothness-weight'] \
                        + network.loss_adversarial(disc_out[1:]) * adversary_weight

                # compute discriminator loss
                disc_loss = network.loss_discriminator(
                    disc_out2[:param['batch-size']], disc_out2[param['batch-size']:])

                total = total + gen_loss.data.numpy() if total is not None else gen_loss.data.numpy()

                average = gen_loss.data.numpy() if average is None else average * \
                    0.95 + 0.05 * gen_loss.data.numpy()
                print('↑' if gen_loss.data.numpy() >
                      average else '↓', end='', flush=True)

                # update parameters step

                gen_loss.backward()
                disc_loss.backward()

                opt_gen.step()
                opt_disc.step()

                # rebuild real discriminator from clone
                enhancer.assign_back_discriminator(disc)

            total /= param['epoch-size']
            stats /= param['epoch-size']

            print('\nOn Epoch: ' + str(epoch))

            print('Generator Loss: ')
            print(total)

            real, fake = stats[:param['batch-size']
                               ], stats[param['batch-size']:]
            print('  - discriminator', real.mean(), len(np.where(real > 0.5)[0]),
                  fake.mean(), len(np.where(fake < -0.5)[0]))

            if epoch == param['adversarial-start'] - 1:
                print('  - generator now optimizing against discriminator.')
                adversary_weight = param['adversary-weight']

            # Then save every several epochs
            if epoch % 10 == 0:
                torch.save(enhancer.state_dict(),
                           'model/model_' + mode + '.pth')

        torch.save(enhancer.state_dict(), 'model/model_' + mode + '.pth')
        print("Training ends after: " +
              str(float(time.time() - start)) + " seconds")
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.exit()

    enhancer = network.Enhancer()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # pretrain network
    if sys.argv[1] == 'pretrain':
        train(enhancer, 'pretrain', pretrain_params)

    # train network
    else:
        enhancer.load_state_dict(torch.load('model/model_pretrain.pth'))
        enhancer.create_new_discriminator(64)

        train(enhancer, 'train', train_params)

# load to cpu --> torch.load('model/model_train.pt', map_location={'cuda:0': 'cpu'})
# for inference, do something like enhancer = network.Enhancer(64)
