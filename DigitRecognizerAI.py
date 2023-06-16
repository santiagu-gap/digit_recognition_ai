#If you want to be able to run this code, you will need to install these libraries through your terminal first.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# # #Getting the MNIST database
mnist = tf.keras.datasets.mnist #

# # #Splitting the data into training and testing data while loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()#

# # #Normalizing the data
x_train = tf.keras.utils.normalize(x_train, axis=1)#
x_test = tf.keras.utils.normalize(x_test, axis=1)#

# # #Creating the model to be sequential
model = tf.keras.models.Sequential()#

# # #Building our own neural network
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))#
model.add(tf.keras.layers.Dense(128, activation='relu'))#
model.add(tf.keras.layers.Dense(128, activation='relu'))#
model.add(tf.keras.layers.Dense(10, activation='softmax'))#

# # #Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])#

# # #Training the model
model.fit(x_train, y_train, epochs=3)#

# # #Saving the model so we don't have to retrain it every time
model.save('handwritten.model')#

#Loading the previously trained model
model = tf.keras.models.load_model('handwritten.model')

#Evaluating the model
loss, accuracy = model.evaluate(x_test, y_test)#
print(loss)#
print(accuracy)#

#---------------------------------------------------------------------------------------------------------------------------------------------------------

#Once you run the code once, you can comment out all the lines that have a '#' after so it loads faster

#---------------------------------------------------------------------------------------------------------------------------------------------------------

#Using the model! (To loop through the numbers, you need to close the tab that displays the number which pop ups when you run the code)
image_number = 1    
while os.path.isfile(f"Samples/Sample{image_number}.png"):
    try:
        img = cv2.imread(f"Samples/Sample{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        
    except:
        print("Error!")
    finally:
        image_number += 1

#The model will print it's prediction on the output!

#---------------------------------------------------------------------------------------------------------------------------------------------------------

#If you want to try out the model yourself:
# 1. Open paint app
# 2. Click on the resize button
# 3. Uncheck the "maintain aspect ratio" button
# 4. Click on the "pixels" button
# 5. Change the values so horizontal and vertical are 28
# 6. Zoom in, and draw a number
# 7. Once you drew the button, click on "file", "save as", and go to the "Samples" folder
# 8. Name the file "Sample" + whatever number comes after the biggest one in the folder
# 9. For example, in my "Samples" folder the biggest number is "Sample3", so I'm going to name the new file "Sample4"
# 10. Once you named the file, click on save, and come test out the AI by running the code! Enjoy :)