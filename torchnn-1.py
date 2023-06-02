import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Resize

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer
clf = ImageClassifier()
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    for epoch in range(10):  # train for 10 epochs
        for batch in dataset:
            X, y = batch
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")

    # Save the model's state_dict
    torch.save(clf.state_dict(), 'model_state.pt')

    # Switch to eval mode for inference
    clf.eval()

    # Load and process the image
    img = Image.open('img_3.jpg')

    # Define a transform pipeline
    transform = transforms.Compose([
        Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        ToTensor()
    ])

    # Apply the transform to the image
    img_tensor = transform(img).unsqueeze(0)

    # Apply the model to the image tensor
    output = clf(img_tensor)

    # Get the predicted class
    predicted_class = torch.argmax(output)

    # Print the predicted class
    print(predicted_class)
