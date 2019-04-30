# Sign Language Detector
A CNN project for detecting indian sign language.
Here, we have implemented CNN (Convolution Neural Network) using Keras a wrapper of Tensorflow.

### Tools Used
1. Python 3
2. OpenCV 3
3. Tensorflow
4. Keras

### Running this project
1. Install Python 3, Opencv 3, Tensorflow, Keras.
2. First Train the model.
    ```
    python cnn_model.py
    ```
2. Now to make use the model you just need to run recognise.py . To do so just open the terminal and run following command.
    ```
    python recognise.py
    ```
    Adjust the hsv values from the track bar to segment your hand color.

3. To create your own data set.
    ```
    python capture.py
    ```
4. For Random Image testing you need to run test_capture.py for capturing images and execute test_model.py for testing the captured images.
    ```
    python test_capture.py
    python test_model.py
    ```
 5. run speech.mp3 file for listening the recognised text from sequence of characters.
    ```
    speech.mp3
    ```
