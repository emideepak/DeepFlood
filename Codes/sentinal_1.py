%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Program:To Identify the Flood/No Flood images using Sentinel 1 images 
Input: Flood and NoFlood VV and VH images mean image.
Output: Flood / No Flood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------------------------------------------------------------------------------
"IMPORT THE LIBRARIES"

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import seaborn as sns
import pandas as pd

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout,Conv2D,MaxPool2D ,BatchNormalization
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


#------------------------------------------------------------------------------------------------------------------------
"IMAGE PREPROCESSING"

SIZE = 75  #Resize images

#Capture training data and labels into respective lists
data_images = []
data_labels = [] 

for directory_path in glob.glob("Dataset/S1/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        # print(img_path)
        img = cv2.imread(img_path, -1)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data_images.append(img)
        data_labels.append(label)

#Convert lists to arrays        
data_images = np.array(data_images)
data_labels = np.array(data_labels)



#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data_labels)
data_labels_encoded = le.transform(data_labels)

# Data Normalization
data_images = data_images / 255.0


img_dir='Dataset/S1'
datagen=ImageDataGenerator(rescale=1/255)


data_gen=datagen.flow_from_directory(img_dir,
                                      target_size=(75,75),
                                      batch_size=4,
                                      class_mode='binary')


#------------------------------------------------------------------------------------------------------------------------
"FEATURE EXTRACTION USING CNN AND VGG16"


"CNN Model Feature Extraction"


model = Sequential()
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (75,75,3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(512 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
# model.add(Dense(units = 10 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy')
model.summary()


history = model.fit(data_images,data_labels_encoded, batch_size = 32 ,epochs = 1 )

"Save the Model"
# model.save("cnn_feat_model_for_s1.h5")


# Load the Saved Feature Extracted Model
# model = load_model("cnn_feat_model_for_s1.h5")

# Model Prediction
x_features_cnn = model.predict(data_images)
x_features_cnn = x_features_cnn.reshape(x_features_cnn.shape[0], -1)


"PLOT THE FEATURE MAP"

def get_outputs(model):
    layer_names=[]
    outputs=[]
    for layer in model.layers:
        if ('conv2d' in layer.name) or ('pooling' in layer.name) :
            layer_names.append(layer.name)
            outputs.append(layer.output)
    return layer_names,outputs


layer_names,outputs=get_outputs(model)

model=Model(model.input,outputs)

def vis_activations(img,model,layer_names):
    activations=model.predict(img)
    images_per_row=16
    for layer_name,activation in zip(layer_names,activations):
        nb_features=activation.shape[-1]
        size=activation.shape[1]
        
        nb_cols=nb_features//images_per_row
        grid=np.zeros((size*nb_cols,size*images_per_row))
        
        for col in range(nb_cols):
            for row in range(images_per_row):
                feature_map=activation[0,:,:,col*images_per_row+row]
                feature_map-=feature_map.mean()
                feature_map/=feature_map.std()
                feature_map*=255
                feature_map=np.clip(feature_map,0,255).astype(np.uint8)
                grid[col*size:(col+1)*size, row*size:(row+1)*size] = feature_map
        scale = 1./size
        plt.figure(figsize=(scale*grid.shape[1], scale*grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.show()
    return activations


img=data_gen.next()[0][0,:,:,:].reshape(-1,75,75,3)
activations=vis_activations(img,model,layer_names)



#------------------------------------------------------------------------------------------------------------------------
"SAVING THE S1 FEATURE"


# np.save("x_features_s1.npy",x_features_cnn)
# x_features_cnn = np.load("x_features_s1.npy")

#------------------------------------------------------------------------------------------------------------------------
"SPLITTING INTO TRAIN AND TEST"


from sklearn.model_selection import train_test_split
# x_train, x_test,y_train,y_test = train_test_split(x_features_cnn,data_labels_encoded,test_size = 0.2)


"Saving the Train and Test Split"
# np.save("S1_Train_Test/x_train.npy",x_train)
# np.save("S1_Train_Test/x_test.npy",x_test)
# np.save("S1_Train_Test/y_train.npy",y_train)
# np.save("S1_Train_Test/y_test.npy",y_test)

"Load the Train and Test Split"

x_train = np.load("S1_Train_Test/x_train.npy")
x_test = np.load("S1_Train_Test/x_test.npy")
y_train = np.load("S1_Train_Test/y_train.npy")
y_test = np.load("S1_Train_Test/y_test.npy")


#------------------------------------------------------------------------------------------------------------------------
"MODEL PREDICTION"

"Using Random Forest"

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 20, random_state = 42)

# Train the model on training data
RF_model.fit(x_train, y_train) 


#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(x_test)



#------------------------------------------------------------------------------------------------------------------------
"PERFORMANCE METRICS"

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print("\n\tACCURACY SCORE\n\t******************************\n")
print (f"\t{accuracy_score(y_test, prediction_RF)*100}")

print("\n\tCLASSIFICATION REPORT\n\t******************************\n")
print (f"\t{classification_report(y_test, prediction_RF)}")


#------------------------------------------------------------------------------------------------------------------------
"CONFUSION MATRIX"

cm = confusion_matrix(y_test,prediction_RF)
fig, ax = plt.subplots(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Flood', 'No Flood']); ax.yaxis.set_ticklabels(['Flood', 'No Flood']);


#------------------------------------------------------------------------------------------------------------------------
"RF ESTIMATORS PLOT"


estimators = [ '2' , '5' , '10' , '20' ,
        '25' , '50' , '100' ]
  
Accuracy = [ 73.91 , 80.43 , 82.60 , 82.60 ,
            82.60 , 82.60 , 82.60 ]
  
df_estimators = pd.DataFrame(
    { 'estimators' : estimators , 'Accuracy' : Accuracy })
df_estimators.plot( 'estimators' , 'Accuracy' )

