**Second Sight** is an Android app that can read text "in the wild" (e.g. price tags, product labels, text on shirts) using TensorFlow and Google Cloud Vision. It's my Udacity Machine Learning Nanodegree capstone project.

## Features
* Detect text using the camera
* Read detected text (requires an Internet connection)
* Interrupt speech by touching the lower portion of the screen

Check out the Conclusion section of the [Project Report](https://github.com/martinbede/second-sight/blob/master/Project_Report.pdf) for a description of the overall performance.

## How to install
A prebuilt APK is available in the bin directory: [download](https://github.com/martinbede/second-sight/blob/master/bin/second-sight.apk)

The following instructions are from the TensorFlow Android Demo README:

If adb debugging is enabled on your Android 5.0 or later device, you may then
use the following command from your workspace root to install the APK once
built:
 
```bash
$ adb install -r -g bazel-bin/second-sight/second-sight.apk
```
 
Some older versions of adb might complain about the -g option (returning:
"Error: Unknown option: -g").  In this case, if your device runs Android 6.0 or
later, then make sure you update to the latest adb version before trying the
install command again. If your device runs earlier versions of Android, however,
you can issue the install command without the -g option.

## Tips for modifying
The neural network running on the device and detecting text is trained using [Jupyter Notebooks](http://jupyter.org/) using the following Python libraries:
* NumPy 1.10.4
* Python Imaging Library 1.1.7
* MatPlotLib 1.4.3
* TensorFlow 0.7.1
* SciPy 0.16.0

These are the most important files of the project: 
* notebooks/Prepare data.ipynb: Downloads the datasets and prepares the data for training
* notebooks/Create and freeze graph.ipynb: Defines, trains, and saves the classifier, which is a ConvNet that looks for text in images
* second-sight/src/.../TensorflowImageListener.java: Preprocesses images, calls the JNI code to do inference, calls the Cloud Vision API, etc.
* second-sight/jni/tensorflow_jni.cc: The inference happens here

The notebooks use bash commands to create/delete directories and to download and unzip archives. These might not run on some systems (e.g. Windows). In that case, you have to do those tasks manually.

Also note that training the neural network requires a computer with a relatively new NVIDIA GPU (e.g. GeForce 980TI), at least 8 GB of RAM, and a fast internet connection (the dataset used for training is larger than 10 GB).

## Build instructions
1. Make sure you have the [Android SDK](http://developer.android.com/sdk/index.html) (API level **23**, sdk-build-tools version **23.0.1**), NDK **r10e** (see download links below), [TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html) (for Python 2.7), and [Bazel **0.1.5**](http://bazel.io/docs/install.html) installed (It's strongly recommended to use Ubuntu or OS X for building the application, as the Windows support of Bazel is experimental)
2. Get the project from GitHub `git clone https://github.com/martinbede/second-sight --recursive` 
3. Modify the WORKSPACE file so that the paths to the Android SDK and NDK are correct
4. Get a Google Cloud Vision key and create a file at second-sight/res/values/keys.xml with the following content:
```
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="CloudVisionApiKey">ENTER_KEY_HERE!!</string>
</resources>
```
5. Build the project `bazel build //second-sight:second-sight`

To build, install, and run the app: `bazel mobile-install //second-sight:second-sight --start_app`

If you get build errors about protocol buffers, run
`git submodule update --init` and build again.

### NDK r10e download links
* [Linux x86](http://dl.google.com/android/ndk/android-ndk-r10e-linux-x86.bin)
* [Linux x86_64](http://dl.google.com/android/ndk/android-ndk-r10e-linux-x86_64.bin)
* [OS X x86_64](http://dl.google.com/android/ndk/android-ndk-r10e-darwin-x86_64.bin)
* [Windows x86](http://dl.google.com/android/ndk/android-ndk-r10e-windows-x86.exe)
* [Windows x86_64](http://dl.google.com/android/ndk/android-ndk-r10e-windows-x86_64.exe)

## Used works
* [TensorFlow numerical computation library](https://www.tensorflow.org/)
* [Microsoft Common Objects in Context dataset](http://mscoco.org/)
* [COCO-Text dataset](http://vision.cornell.edu/se3/coco-text/)
* [Google Cloud Vision API](https://cloud.google.com/vision/)
