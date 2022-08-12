# VGG-Lite
This project implemented a convolutional neural network (CNN) based on the VGGNet architecture to classify images of cats and dogs. The purpose of this project was to practice creating, implementing, and fine-tuning CNNs. The goal was to achieve the highest accuracy possible within a two week period. The results were 70% training accuracy and 68% testing accuracy.

## Dataset
The dataset used consisted of 4800 images of cats and dogs. The dataset was split into three unique subsets of training, validation, and testing. The breakdown of the images was as follows:
* Training = 4000 images (2000 cats, 2000 dogs)
* Validation = 600 images (300 cats, 300 dogs)
* Testing = 200 imagse (100 cats, 100 dogs)

## Hyperparameters
The final hyperparameters of the model are as follows:
* Epochs = 100
* Batch size = 32
* Training samples = 4000
* Validation samples = 600
* Testing samples = 200
* Image width = 224
* Image height = 224
* Channels = 3
* Loss function = Binary cross entropy
* Optimizer = SGD

## Network
Several "VGG-Lite" models of different depths were tested before fine-tuning. The different models were based on VGG11, VGG16, and VGG19. A "VGG9" was also created by removing a chunk of convolutions from VGG11. As VGGNet was designed to classify 1000 images, the final layers of the models were altered to fit the classification of two classes instead. The final layers of each of the models replaced the original FC-1000 and Softmax layers with FC-1 and Sigmoid layers. The loss function used was binary cross entropy loss, since there were only two classes. The optimizer used was stochastic gradient descent (SGD), with learning rate and momentum set to 0.01 and 0, respectively. The hyperparameters used to train the models were the same as listed above except the number of epochs 50. The details of the models can be found in the VGGNet paper published in 2014 by Simonyan and Zisserman (https://arxiv.org/abs/1409.1556).

## Results
The loss and accuracy of each model during training were compared. The general pattern was, as the amount of layers increased, the loss increased and the accuracy decreased. As the "VGG9" version did the best overall, it was choses as the model to fine-tune, using the final hyperparameters listed above. The final results after training the "VGG9" model showed highs of 70.45% training accuracy, 73.78% validation accuracy, and 67.50% testing accuracy.

## Comments/Future Works
Due to the time restriction and limited computing resources, I was only able to experiment with the different-depth VGG models and number of epochs. I hope to experiment at a future date the effects of different loss functions and optimizers, learning rates and momentums, and more recent architectures, including GoogLeNet and ResNet.
