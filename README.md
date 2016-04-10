**Second Sight** is an Android app that can read text "in the wild" (e.g. pricetags, product labels, text on shirts) using TensorFlow and Google Cloud Vision. It's my Udacity Machine Learning Nanodegree capstone project.

## Features
* Detect text using the camera
* Read detected text (requires an Internet connection)
* Interrupt speech by touching the lower portion of the screen

### Examples of text the app can read
(TODO)

### Examples of text the app can't read
(TODO)

## Building
1. Make sure you have the [Android SDK](http://developer.android.com/sdk/index.html) (API level 23), [NDK](http://developer.android.com/ndk/downloads/index.html) (API level 21), [TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html) (for Python 2.7), and [Bazel](http://bazel.io/docs/install.html) installed (It's recommended to use Ubuntu for building the application, as the Windows support of Bazel is experimental)
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

## Installing
From the TensorFlow Android Demo README:

If you get build errors about protocol buffers, run
`git submodule update --init` and build again.

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

Alternatively, a streamlined means of building, installing and running in one
command is:

```bash
$ bazel mobile-install //second-sight:second-sight --start_app
```

If camera permission errors are encountered (possible on Android Marshmallow or
above), then the `adb install` command above should be used instead, as it
automatically grants the required camera permissions with `-g`. The permission
errors may not be obvious if the app halts immediately, so if you installed
with bazel and the app doesn't come up, then the easiest thing to do is try
installing with adb.

## Modifying
These are the most important files of the project: 
* notebooks/Prepare data.ipynb: Downloads the datasets and prepares the data for training
* notebooks/Create and freeze graph.ipynb: Defines, trains, and saves the classifier, which is a ConvNet that looks for text in images
* second-sight/src/.../TensorflowImageListener.java: Preprocesses images, calls the JNI code to do inference, calls the Cloud Vision API, etc.
* second-sight/jni/tensorflow_jni.cc: The inference happens here

## Used works
* [TensorFlow numerical computation library](https://www.tensorflow.org/)
* [Microsoft Common Objects in Context dataset](http://mscoco.org/)
* [COCO-Text dataset](http://vision.cornell.edu/se3/coco-text/)
* [Google Cloud Vision API](https://cloud.google.com/vision/)
