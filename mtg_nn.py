import torch
from torchvision import transforms, datasets
import json
from collections import defaultdict
import os
import random
import requests
import shutil
import ssl
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class Params:

	classes = ['B','Colorless','G',"Gold","R",'U','W'] #the single letters stand for Black, Green, Red, Blue, and White respectively.
	#All multi-colored cards are referred to here as gold

	count = True #exploratory data analysis
	makedirs = True #make the directories to download into
	download = True #download the images
	train = True #train the model
	test = True #test the model on the test set

	epochs = 30 #number of epochs to train for
	show = 3 #number of images to plot at test

class myCNN(nn.Module):

	#A relatively basic CNN architecture
	#4 convolutional layers, each followed by a 2x2 max pooling
		#3x3 filter and padding 1 in each layer keeps size consistent
	#followed by 3 fully-connected layers
		#Dropout is used during training as a form of regularization
	#while I haven't done a proper validation step for this architecture, I found tried a few sizes and this one gave the best test accuracy

		def __init__(self, num_classes):
			super().__init__()
			self.conv_block = nn.Sequential(
				nn.Conv2d(3,64,kernel_size=3,padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2,2),

				nn.Conv2d(64,64,kernel_size=3,padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2,2),

				nn.Conv2d(64,128,kernel_size=3,padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2,2),

				nn.Conv2d(128,128,kernel_size=3,padding=1),
				nn.ReLU(),
				nn.MaxPool2d(2,2),
				)
			self.fc_block = nn.Sequential(
				nn.Dropout(0.5),
				nn.Linear(128*8*8,512),
				nn.ReLU(),
				nn.Linear(512,128),
				nn.ReLU(),
				nn.Linear(128,7)
				)
		def forward(self,x):
			x = self.conv_block(x)
			x = x.view(x.size(0),-1)
			x = self.fc_block(x)
			return x

def count():

	p = Params()

	with open('unique-artwork-20250409090418.json') as jf: #the data for this is the set of all unique artworks
		cards = json.load(jf)
		types = defaultdict(int)
		
		for card in cards:
			#there are several things to exclude here, namely cards with poorly defined color (like tokens and lands) as well as cards with more than one artwork (double-sided cards)
			if card['layout'] == "reversible_card" or card['layout'] == "transform" or card['layout'] == "modal_dfc" or card['layout'] == "double_faced_token" or card['set_type'] == 'memorabilia' or "Token" in card["type_line"] or "Land" in card["type_line"] or card["oversized"] == True:
				continue

			colors = card['colors']
			if len(colors) == 0:
				types['Colorless'] += 1
			elif len(colors) > 1:
				types['Gold'] += 1
			else:
				types[colors[0]] += 1

		print(types)
		print(sum(types.values()))
		print(len(cards))

def dirs():

	p = Params()

	for color in p.classes:
		d = "test/"+color
		if not os.path.exists(d):
			os.makedirs(d)
		d = "train/"+color
		if not os.path.exists(d):
			os.makedirs(d)


def download():

	p = Params()

	with open('unique-artwork-20250409090418.json') as jf: #see the discussion in the count section
		cards = json.load(jf)

	file_names = set([])

	for card in cards:
		if card['layout'] == "reversible_card" or card['layout'] == "transform" or card['layout'] == "modal_dfc" or card['layout'] == "double_faced_token" or card['set_type'] == 'memorabilia' or "Token" in card["type_line"] or "Land" in card["type_line"] or card["oversized"] == True:
				continue

		dest = "train"
		if random.random() > 0.8: #80/20 test/train split
			dest = "test"

		colors = card['colors'] #determine labels as directory names
		if len(colors) == 0:
			color = "Colorless"
		elif len(colors) > 1:
			color = 'Gold'
		else:
			color = colors[0]

		file_name = card['name'].replace(" ","_").replace("//",'SPLIT')+".jpg" #// represents split cards
		count = 2
		while file_name in file_names: #since we are allowing for multiple copies of a card if each copy has different artwork, we need a better way of naming the cards
			file_name = str(count)+"_"+file_name #this does not work the way I originally intended (see sol ring 32_31_..._2_sol_ring.png) but is good enough
			count += 1
		path = dest+"/"+color+"/"+file_name

		response = requests.get(card["image_uris"]["art_crop"]) #importantly, we are only working with the art crop here, not the full card

		if response.status_code == 200:
			with open(path, "wb") as f:
				f.write(response.content) #write the downloaded image
				file_names.add(file_name)
			print("Got",file_name)
		sleep(0.1) #so as to not overload the server

def train():

	p = Params()

	#loaded images are converted to 128x128. There is a 50% chance they are mirrored. They are converted to PyTorch tensors, scaled and normalized
	#128x128 seems to be a little bigger than the standard, but i'm hoping it helps 
	transform_train = transforms.Compose([transforms.Resize((128, 128)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

	train_dataset = datasets.ImageFolder(root="train", transform=transform_train) #the file structure is where it gets the labels

	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #this dataset isn't so big that a batch size of 64 is prohibitive
			
	if torch.backends.mps.is_available(): #I have a m3 macbook, which PyTorch supports. 
		device = torch.device("mps")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_classes = len(train_dataset.classes)

	model = myCNN(num_classes=num_classes).to(device)


	criterion = nn.CrossEntropyLoss() #So, I don't do a validation step here for hyperparameter tuning, but if I was using this for something important I would want to test these choices
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	def train_model(model, loader, criterion, optimizer, epochs=p.epochs): #I hate that PyTorch makes you do this yourself, btw
	    
	    for epoch in range(epochs): #i want to say, for the record, that I find the epoch thing weird
    	#like in my field we do a lot of complex numerical optimization
    	#but we use a convergence threshold
    	#and, like, more complex optimization algorithms than gradient descent
    	#i guess the data here is too big for that to work? Or is it to avoid overfitting? If you're reading this and you have a better answer, let me know
    	#anyway, I chose 30 epochs because I saw an example online that used 30

	    	model.train() #train mode
	    	total_loss, correct, total = 0, 0, 0

	    	for images, labels in loader: #loop over your mini-batches

	    		images, labels = images.to(device), labels.to(device)
	    		optimizer.zero_grad() #zero out your gradients from the last iteration
	    		outputs = model(images) #make predictions
	    		loss = criterion(outputs, labels) #compute loss
	    		loss.backward() #back-propagation for gradients
	    		optimizer.step() #update weights

	    		total_loss += loss.item() #for keeping track of how we're doing
	    		_, predicted = torch.max(outputs, 1) #this turns predicted vectors into classification labels
	    		total += labels.size(0)
	    		correct += (predicted == labels).sum().item()
    		print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {100*(correct/total):.2f}%") #train accuracy is obviously not the important thing here but its good to know


	train_model(model, train_loader, criterion, optimizer)

	torch.save(model.state_dict(),'./mtg_net.pth') #save your weights so you don't have to re-train every tune

	return model

def test(model=False):

	p = Params()
	if not model: #if you trained and tested in the same code, it doesn't need to read in a new model from a saved file
		model = myCNN(num_classes=len(p.classes))

		model.load_state_dict(torch.load('./mtg_net.pth'))

	if torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	model.eval() #put the model in test mode

	transform_test = transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]) #load the test data and transform it appropriately
	test_dir = "test"
	test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
	test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

	def test_model(model, loader, num_shown = p.show): #again, it feels like there should be an easier way to do this
		correct, total = 0, 0
		all_preds = []
		all_labels = []
		shown = 0

		with torch.no_grad(): #even though you're in test mode, you have to tell it again not to store gradients
			for images, labels in loader: #mini-batches. why not?
				images, labels = images.to(device), labels.to(device)
				outputs = model(images) #make predictions
				_, predicted = torch.max(outputs, 1) #get labels as the max probability

				total += labels.size(0)
				correct += (predicted == labels).sum().item() #for test accuracy

				all_preds.extend(predicted.cpu().numpy()) #for confusion matrix
				all_labels.extend(labels.cpu().numpy())

				if shown < num_shown: #show a few examples for funsies

					plt.figure(figsize=(3, 3))
					img = images[0].cpu()
					img = img.numpy().transpose((1, 2, 0)) #numpy and pytorch like the RGB in different places
					img = img * np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5)) #reverse transform
					img = np.clip(img, 0, 1) #ensures all values are between 0 and 1
					plt.imshow(img)
					plt.axis('off')
					plt.title(f"Predicted: {p.classes[predicted[0]]}\nActual: {p.classes[labels[0]]}")
					plt.show()
					shown += 1
                

		print(f"Test Accuracy: {100*correct/total:.2f}%") #print test accuracy

		cm = confusion_matrix(all_labels,all_preds) #plot confusion matrix
		disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=p.classes)
		disp.plot(cmap=plt.cm.Blues)
		plt.xticks(rotation=45)
		plt.title("Confusion Matrix")
		plt.show()

	test_model(model, test_loader)


def main():

	p = Params()

	if p.count:
		count()

	if p.makedirs:
		dirs()

	if p.download:
		download()

	if p.train:
		model=train()

	if p.test:
		if p.train:
			test(model)
		else:
			test()

if __name__ == '__main__':
    main()