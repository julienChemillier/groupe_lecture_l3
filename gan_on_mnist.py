import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt



# Setting up the parameters for the GANs
cuda_usage = True
DATA_PATH = './data'
BATCH_SIZE = 128
NB_IMAGE_CHANNEL = 1
Z_DIM = 100
HIDDEN = 64
LEAKY_LEAKER = 0.2
X_DIM = 64
NB_EPOCH = 5
REAL_LABEL = 1
FAKE_LABEL = 0
learning_rate = 2e-4
betas = (0.5, 0.999)
seed = 42


# Setting up CUDA for GPU usage if available (mainly used in google colab)
cuda_available = cuda_usage and torch.cuda.is_available()
print("PyTorch version:", torch.__version__)
if cuda_available:
    print("CUDA version: ", torch.version.cuda, "\n")

if cuda_available:
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
cudnn.benchmark = True


# Loading MNIST dataset
dataset = dset.MNIST(root=DATA_PATH, download=True, transform=transforms.Compose([transforms.Resize(X_DIM), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Show the first 64 training data
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))



# -----------------------------------------------------
# ----------------------Classes------------------------
# -----------------------------------------------------

def weights_init(m):
    ''' Initialize the weights of m according its layer type'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    ''' Classe for the generator neural network that inherits from the Module class'''
    def __init__(self):
        # We uses the Module not to redo the all intialization
        super(Generator, self).__init__()

        # Setting up its different layers
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(HIDDEN * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(HIDDEN * 8, HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(HIDDEN * 4, HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(HIDDEN * 2, HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN),
            nn.ReLU(True),
            nn.ConvTranspose2d(HIDDEN, NB_IMAGE_CHANNEL, 4, 2, 1, bias=False),
            # We choose this activation to have pixels in [-1, 1]
            nn.Tanh()
        )

    def forward(self, input):
        ''' Method to launch the forward propagation'''
        return self.main(input)


class Discriminator(nn.Module):
    ''' Classe for the discriminator neural network that inherits from the Module class'''
    def __init__(self):
        # We uses the Module not to redo the all intialization
        super(Discriminator, self).__init__()

        # Setting up its different layers
        self.main = nn.Sequential(
            nn.Conv2d(NB_IMAGE_CHANNEL, HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(LEAKY_LEAKER, inplace=True),
            nn.Conv2d(HIDDEN, HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN * 2),
            nn.LeakyReLU(LEAKY_LEAKER, inplace=True),
            nn.Conv2d(HIDDEN * 2, HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN * 4),
            nn.LeakyReLU(LEAKY_LEAKER, inplace=True),
            nn.Conv2d(HIDDEN * 4, HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN * 8),
            nn.LeakyReLU(LEAKY_LEAKER, inplace=True),
            nn.Conv2d(HIDDEN * 8, 1, 4, 1, 0, bias=False),
            # We choose this activation to have probabilities in [0, 1]
            nn.Sigmoid()
        )

    def forward(self, input):
        ''' Method to launch the forward propagation'''
        return self.main(input).view(-1, 1).squeeze(1)



# -----------------------------------------------------
# ----------------------Training-----------------------
# -----------------------------------------------------

# Creation and initialization of both generator and discriminator networks
generator = Generator().to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

# Setting up for the backpropagation
loss_function = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)

# Setting up variables to follow evolution thourgh epochs
img_list = []
loss_generator = []
loss_discriminator = []
iterations = 0
fixed_64_noise = torch.randn(64, Z_DIM, 1, 1, device=device)

for epoch in range(NB_EPOCH):
    for i, data in enumerate(dataloader, 0):
        # Discriminator with real data
        discriminator.zero_grad()
        real_adapted_device = data[0].to(device)
        real_size = real_adapted_device.size(0)
        label = torch.full((real_size,), REAL_LABEL, dtype=torch.float, device=device)
        output = discriminator(real_adapted_device).view(-1)
        discriminator_error_real = loss_function(output, label)
        discriminator_error_real.backward()
        D_x = output.mean().item()

        # Discriminator with fake data
        noise = torch.randn(real_size, Z_DIM, 1, 1, device=device)
        fake_data = generator(noise)
        label.fill_(FAKE_LABEL)
        output = discriminator(fake_data.detach()).view(-1)
        discriminator_error_fake = loss_function(output, label)
        discriminator_error_fake.backward()
        D_G_z1 = output.mean().item()
        errD = discriminator_error_real + discriminator_error_fake
        optimizerD.step()

        # Generator with fake data
        generator.zero_grad()
        label.fill_(REAL_LABEL)
        output = discriminator(fake_data).view(-1)
        errG = loss_function(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Save the error
        loss_generator.append(errG.item())
        loss_discriminator.append(errD.item())

        # Display the training evolution
        """if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, NB_EPOCH, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))"""

        iterations += 1

    # At each epoch plot 64 real images to compare ...
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # ... with fake images generated
    # We make sure to keep the same noise so that we can compare from one epoch to another
    fake_images = generator(fixed_64_noise).detach().cpu()
    img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
    img_list.append(img_grid)
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
