# Visionary android app
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

### Screenshots
