# Cat_vs_Dog_Classifier-CNN-

This project implements a deep learning model (inspired by VGG16 with Batch Normalization) to classify image of cats and dogs.  

This projects implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. It is trained on the Kaggle Dogs vs Cats dataset
 and achieves high validation accuracy of (0.9486) and val_loss of (0.1730) in distinguishing between the two classes. (Link: https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)

WEIGHTS LINK --> The cat_dog_model.weights.h5 has been uploaded to drive (Link: https://drive.google.com/file/d/1Ig-8CdwBcmDX6Y0dd33nwlYAhz8_ql3d/view?usp=sharing) 

The model is trained to automatically learn features from the images and predict whether a given input image belongs to the "Cat" class or the "Dog" class.
This project is implememted in Python using Tensorflow and Keras. The training proccess was performed on Goolgle Colab, with some acceralation GPU to speed up computations. To prevent model from overfitting and improve generalization, tequniques such as Dropout, BatchNormalization and early stopping were applied.
ModelCheckpoint was also been in pratice in order to get the best weights.

The final model achieved good accuracy on the validation dataset and is capable of predicting unseen images with high confidence. 
To run the project, one needs to clone the repository, install the dependencies listed in requirements.txt, and either train the model again or directly load the saved best weights (best_model.weights.h5).

This project demonstrates the application of deep learning in computer vision, specifically in image classification tasks. 
It can serve as a foundation for more complex projects such as multi-class animal classification or deployment of the model into a web application using tools like Gradio or Streamlit.

Model Architecture: Saved in Cat_vs_Dog_Classification.ipynb ......
