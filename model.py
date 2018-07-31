import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import json
# Create a customized classifier builder
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p = 0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def save_checkpoint(model, optimizer, image_dataset, epochs, file_path = None, file_name = 'checkpoint.pth', print_model = False):
    # TODO: Save the checkpoint 
    if file_path is not None:
        file_name = file_path + '/' + file_name

    model.class_to_idx = image_dataset.class_to_idx
    #model.cat_to_name = cat_to_name
    model.epochs = epochs + 1
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'class_to_idx': model.class_to_idx,
                  #'cat_to_name': model.cat_to_name,
                  'epochs': model.epochs,
                  'state_dict': model.classifier.state_dict(),
                  'optimizer' : optimizer.state_dict()
                 }
    if print_model:
        print(model.classifier)
    torch.save(checkpoint, file_name)
    print('Checkpoint file has been saved to: ', file_name)


def load_checkpoint(file_path = None, file_name = 'checkpoint.pth'):
    if file_path is not None:
        file_name = file_path + '/' + file_name
    checkpoint = torch.load(file_name, map_location={'cuda:0': 'cpu'})
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = Classifier(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], drop_p = 0.2)
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    #model.cat_to_name = checkpoint['cat_to_name']
    model.epochs = checkpoint['epochs']
    return model

def load_cat_json(file_path = None, file_name = 'cat_to_name.json'):
    if file_path is not None:
        file_name = file_path + '/' + file_name
    with open(file_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def accuracy_score(model, testloader, device = 'cpu', print_score = True):
    # turn off drop out in the network
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    percent_accuracy = 100 * correct / total
    if print_score:
        print('Accuracy of the network on the {} input images: {}%'.format(total, percent_accuracy))
    return percent_accuracy