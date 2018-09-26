# Visionary android app
- This uses computer vision to identify objects in their vicinity and inform the user using speech. It performs object detection, face detection, emotion recognition, etc. in real time.

### Core components:
- TensorFlow Lite Object detection API
- SSD_Mobilenet_V2_1.0_224 convolutional neural network
- Firebase ML kit text recognition
- Microsoft Cognitive Face API service

### Getting started (Ubuntu 16.04)
You can download the .apk file in the releases section of this GitHub repository. The latest successful build was performed using Android Studio 3.2.

- Install Java and Android Studio: https://medium.com/@aashimad1/install-android-studio-in-ubuntu-b8aed675849f
- Create a folder and run the command: $ git clone https://github.com/himanshurawlani/Visionary
- Open the cloned project in android Studio. Install missing platforms and tools and sync project.
- If prompted to install CMAKE and NDK automatically then try installing it (Note: This would install latest version of NDK). If the build still fails try manual installation of NDK.
- Manually install the following in Android Studio:
    Install CMAKE and NDK in Android studio (SDK Manager -> SDK Tools -> Select the above two tools and Install).
    The latest version of NDK with which this app was successfully built was : 18.0.5002713.
    Ignore the errors “Cannot resolve corresponding JNI function” for all the native methods.
- Add Firebase to your Android Project. To add this app to your Firebase project, use the applicationId value specified in the app/build.gradle file of the app as the Android package name. Download the generated google-services.json file, and copy it to the app/ directory of the sample you wish to run.
- Insert your API key in app/src/main/res/values/secrets.xml. Make sure you add this file to .gitignore file, else your API key would be public on GitHub.
- Install the app on an Android device by clicking run icon.
- Envision the present !

### Important links
- The app was built upon my previous work from this commit: https://github.com/himanshurawlani/SeeFood/tree/6d57be210745811a02ce20a0497c1d458f5d2663
- Hence, you’ll find the package name com.example.ubuntu.seefood. Don’t change the package name else the JNI would break and the app wouldn’t build.
- The TFLiteObjectDetectionAPIModel.java is taken from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/examples/android/app
- The assets folder contains the detection model detect.tflite. The labels of the models are stored in coco_labels_list.txt. Building a TFlite model is another miniproject in itself.
- Pick a Mobilenet model from here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md. I have picked ssdlite_mobilenet_v2_coco for this project.
- Clone the tensorflow repository, install the annoying bazel and other dependencies. Then follow: https://github.com/himanshurawlani/mobilenet_v2_models/blob/master/ssdlite_mobilenet_v2_coco_2018_05_09/tflite_files/Commands_for_converting_to_tflite


### Screenshots
<img src="https://user-images.githubusercontent.com/16475754/46054976-5e23d900-c167-11e8-858b-9711e711217a.png" width="360" height="640">
<img src="https://user-images.githubusercontent.com/16475754/46054977-5e23d900-c167-11e8-8cf5-1ebb0ee41c20.png" width="360" height="640">
<img src="https://user-images.githubusercontent.com/16475754/46054979-5ebc6f80-c167-11e8-8878-c4ac2a7ca106.png" width="360" height="640">
