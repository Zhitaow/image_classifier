import argparse
from torchvision import models
from torch import nn, optim
import torch
import utility
from utility import loadData, process_image
import model
from model import load_checkpoint, load_cat_json
from PIL import Image
import numpy as np

def predict(image_path, model, topk=5, cat_to_name = None, device = 'cpu', image_show = False, probs_show = False, barh_show = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file    
    image = torch.tensor(process_image(Image.open(image_path)))

    image_input = image.resize(1, image.size()[0], image.size()[1], image.size()[2]).float()

    # TODO: Calculate the class probabilities (log_softmax) for img
    
    model.eval()
    model.to(device)
    image_input.to(device)

    class_to_idx = model.class_to_idx
    with torch.no_grad():
        output = model.forward(image_input.to(device))
        _, predicted_idx = torch.topk(output, topk, dim = 1)
        predicted_idx = predicted_idx.cpu().numpy()[0]
    ps = torch.exp(output)
    probs = ps.cpu().numpy()[0, predicted_idx]
    
    # Map the probility-ranked index to the class name
    classes, class_names = [], []
    i = 0
    for idx in predicted_idx:
        for class_id, class_idx in model.class_to_idx.items(): 
            if idx == class_idx:
                i = i + 1
                if cat_to_name is not None:
                	class_name = cat_to_name.get(str(class_id))
                	class_names.append(class_name)
                else:
                	class_names.append(class_id)
                classes.append(class_id)

    classes, class_names = np.array(classes), np.array(class_names)
    
    # Print the classnames, probilities of the top k classes
    if probs_show:
        for i in np.arange(len(class_names)):
            print('Rank {}: Class Name: {}, Probability: {} %'.format(i+1, class_names[i], f'{probs[i]* 100:.2f}'))
            
    # Plot the image and probabilities
    title = class_names[0] + ': ' + f'{probs[0]*100:.1f}'  + '%'
    if (image_show is True) & (probs_show is False):
        ax = imshow(image.numpy())
        ax.set_title(title)
    elif (image_show is True) & (barh_show is True):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1 = imshow(image.numpy(), ax = ax1)
        ax1.set_title(title)
        ax2 = fig.add_subplot(122)
        x = np.arange(topk)
        ax2.barh(x, probs, tick_label = class_names)
        ax2.yaxis.tick_right()
        ax2.figure.set_size_inches(10, 5)
        ax2.set_yticklabels(class_names, fontsize=12)

    
    # return a list of top k class names and their probabilities
    return probs, classes, class_names

def Main():
	parser = argparse.ArgumentParser()
	parser.add_argument("image", help = "The input image to be predicted.", type = str)
	parser.add_argument("checkpoint", help = "Use a mapping of categories to real names", type = str)
	parser.add_argument("--gpu", help = "Use GPU instead of CPU.", action = "store_true")
	parser.add_argument("--topk", help = "Return top K most likely classes. ", type = int, default = 1)
	parser.add_argument("--category_names", help = "Use a mapping of categories to real names", type = str, default = None)    
	args = parser.parse_args()

	if args.gpu:
		device = 'cuda'
		print('Compute using GPU')
	else:
		device = 'cpu'
		print('Compute using CPU')

	checkpoint_path, checkpoint_name = None, args.checkpoint #'checkpoint.pth'
	image_path = args.image #'flowers/test/1/image_06743.jpg'
	topk = args.topk
	category_names = args.category_names #'cat_to_name.json'

	model = load_checkpoint(file_path = checkpoint_path, file_name = checkpoint_name)
	cat_to_name = None
	if category_names is not None:
		cat_to_name = load_cat_json(file_name = category_names)
	
	probs, classes, class_name = predict(image_path, model, topk = topk, cat_to_name = cat_to_name, device = device, probs_show = True)


if __name__ == '__main__':
	Main()  