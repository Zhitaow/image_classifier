import argparse
#import utility
from utility import loadData
import model
from model import Classifier, accuracy_score, save_checkpoint
from torchvision import models
from torch import nn, optim

def train(model, trainloader, testloader, criterion, optimizer, epochs = 10, print_every = 40, print_accuracy = False, device = 'cpu'):
    steps = 0
    # turn on drop out in the network (default)
    model.train()
    # change to device
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                #print("Epoch: {}/{}... ".format(e+1, epochs),
                #      "Loss: {:.4f}".format(running_loss/print_every))
                #running_loss = 0
                if print_accuracy:
                    train_accuracy = accuracy_score(model, trainloader, device = device, print_score = False)
                    test_accuracy = accuracy_score(model, testloader, device = device, print_score = False)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every), "Train accuracy: {} %".format(f'{train_accuracy:.1f}'), "Test accuracy: {} %".format(f'{test_accuracy:.1f}'))
                else:
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0

# def set_optimizer(method = 'Adam', learn_rate = 0.001):
#     optimizer = None
#     if method == 'Adam':
#         optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
#     return optimizer

# def set_criterion(method = 'NLLLoss'):
#     criterion = None
#     if method == 'NLLLoss':
#         criterion = nn.NLLLoss()
#     return criterion


def DenseNet121():
    return models.densenet121(pretrained = True), 1024

def VGG16():
    return models.vgg16(pretrained = True), 25088

def AlexNet():
    return models.alexnet(pretrained = True), 9216

model_options = {
        0: DenseNet121,
        1: VGG16,
        2: AlexNet}

def build_model(model_name = 1, output_unit = 102, hidden_layer = [1], drop_p = 0.2, print_model = False):
    #if model_name == 'vgg16':
    #    model = models.vgg16(pretrained=True)

    model, input_unit = model_options[model_name]()

    if print_model:
        print('Explore the model structure: \n', model)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    # Customize the classifier
    model.classifier = Classifier(input_unit, output_unit, hidden_layer, drop_p = drop_p)
    return model

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", help = "The parent directory to containing subfolders of train and test data.", type = str, default = 'flowers')
    parser.add_argument("--save_dir", help = "The directory of checkpoint file to be saved.", type = str, default = None)
    parser.add_argument("--gpu", help = "Use GPU instead of CPU.", action = "store_true")
    parser.add_argument("--learning_rate", help = "Set learning rate for training.", type = float, default = 0.001)
    parser.add_argument("--hidden_units", nargs="+", type = int, default = [1000], \
        help="Set a list of hidden units: e.g. say two hidden layers of 500 and 200 units, the input format: 500 [space] 200")
    parser.add_argument("-e", "--epochs", help = "Set the number of training iterations.", type = int, default = 10)
    parser.add_argument("--skip_accuracy", help = "Skip the validation on training and testing set in each iteration, and reduce the training time.", action = "store_true")
    parser.add_argument("--arch", help = "Pre-trained Model Options: 0: Densenet121, 1: VGG16, 2: AlexNet ", type = int, default = 1)
    
    args = parser.parse_args()

    if args.gpu:
        device = 'cuda'
        print('Compute using GPU')
    else:
        device = 'cpu'
        print('Compute using CPU')

    data_dir = args.data_directory #'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    file_path = args.save_dir
    print('Training data directory: ', train_dir)
    print('Testing data directory: ', test_dir)
    print('Output checkpoint file directory: ', file_path)

    train_image_dataset, trainloader = loadData(train_dir, train = True, batch_size = 128, shuffle = True)
    test_image_dataset, testloader = loadData(test_dir, train = False, batch_size = 128, shuffle = True)
    train_size = len(trainloader.dataset.imgs)
    print('Total number of samples in the train set: ', train_size)

    hidden_layer = args.hidden_units
    model_name = args.arch
    print('Building the model with hidden layer: {}, using pre-trained model: {}'.format(hidden_layer, model_options[model_name]))
    model = build_model(model_name = model_name, hidden_layer = hidden_layer)
    
    epochs = args.epochs
    print('Number of iteration: ', epochs)
    learn_rate = args.learning_rate
    print('Using the learning rate: ',learn_rate)
    if args.skip_accuracy:
        print_accuracy = False
        print('Skip calculating the accuracy during the training.')
    else:
        print_accuracy = True
        print('Will calculate the accuracy during the training. Expected longer training time.')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)

    print('Start training...')
    train(model, trainloader, testloader, criterion, optimizer, epochs = epochs, print_every = 10, print_accuracy = print_accuracy, device = device)
    accuracy_score(model, trainloader, device = device, print_score = True)

    save_checkpoint(model, optimizer, train_image_dataset, epochs, file_path= file_path, file_name = 'checkpoint.pth', print_model = False)

if __name__ == '__main__':
    Main()