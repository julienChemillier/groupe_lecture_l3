import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Configuration
cuda_usage = True
BATCH_SIZE = 24
Z_DIM = 100
HIDDEN = 64
LEAKY_LEAKER = 0.2
X_DIM = 64
NB_IMAGE_CHANNEL = 1
NB_EPOCH = 301
REAL_LABEL = 1
FAKE_LABEL = 0
learning_rate = 2e-4
betas = (0.5, 0.999)
seed = 42

# Setting up CUDA for GPU usage if available (mainly used in google colab)
cuda_available = cuda_usage and torch.cuda.is_available()
print("PyTorch version: ", torch.__version__)
if cuda_available:
    print("CUDA version: \n", torch.version.cuda, "\n")

if cuda_available:
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Loading Olivetti Faces dataset
faces = np.load("olivetti_faces.npy")
faces = (faces - 0.5)*2  # Put pixels in [-1, 1]
faces = np.expand_dims(faces, axis=1)
dataset = data.TensorDataset(torch.tensor(faces, dtype=torch.float32))
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



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
# ----------------------Training------------------------
# -----------------------------------------------------

# Initialize networks
generator_network = Generator().to(device)
generator_network.apply(weights_init)
discriminator_network = Discriminator().to(device)
discriminator_network.apply(weights_init)

# Setting up for the backpropagation
loss_fun = nn.BCELoss()
discriminator_optimizer = optim.Adam(discriminator_network.parameters(), lr=learning_rate, betas=betas)
generator_optimizer = optim.Adam(generator_network.parameters(), lr=learning_rate, betas=betas)

# To see the evolution
vector_see_evolution = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
loss_generator = []
loss_dicriminator = []

for epoch in range(NB_EPOCH):
    for i, data in enumerate(dataloader, 0):
        # Discriminator with real data
        discriminator_network.zero_grad()
        real_adapted_device = data[0].to(device)
        real_size = real_adapted_device.size(0)
        label = torch.full((real_size,), REAL_LABEL, dtype=torch.float, device=device)
        output = discriminator_network(real_adapted_device).view(-1)
        discriminator_error_real = loss_fun(output, label)
        discriminator_error_real.backward()

        # Discriminator with fake data
        noise = torch.randn(real_size, Z_DIM, 1, 1, device=device)
        fake_data = generator_network(noise)
        label.fill_(FAKE_LABEL)
        output = discriminator_network(fake_data.detach()).view(-1)
        discriminator_error_fake = loss_fun(output, label)
        discriminator_error_fake.backward()
        discriminator_optimizer.step()

        # Generator with fake data
        generator_network.zero_grad()
        label.fill_(REAL_LABEL)
        output = discriminator_network(fake_data).view(-1)
        generator_error_fake = loss_fun(output, label)
        generator_error_fake.backward()
        generator_optimizer.step()

    # Save the error
    loss_generator.append(generator_error_fake.item())
    loss_dicriminator.append(discriminator_error_real.item() + discriminator_error_fake.item())

    # Display images every 25 epochs during training
    with torch.no_grad():
        fake_data = generator_network(vector_see_evolution).detach().cpu()
    img_grid = vutils.make_grid(fake_data, padding=2, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title("Epoch "+ str(epoch))
    plt.axis('off')
    plt.show()
