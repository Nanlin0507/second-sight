/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.martinbede.secondsight;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Handler;
import android.os.Trace;
import android.os.AsyncTask;

import junit.framework.Assert;

import com.martinbede.secondsight.env.ImageUtils;
import com.martinbede.secondsight.env.Logger;

import java.util.List;
import java.util.ArrayList;
import java.lang.Math;
import java.io.*;
import android.content.Context;
import android.util.Base64;

import com.google.api.client.extensions.android.http.AndroidHttp;
import com.google.api.client.googleapis.json.GoogleJsonResponseException;
import com.google.api.client.http.HttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.vision.v1.Vision;
import com.google.api.services.vision.v1.VisionRequestInitializer;
import com.google.api.services.vision.v1.model.AnnotateImageRequest;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesRequest;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesResponse;
import com.google.api.services.vision.v1.model.EntityAnnotation;
import com.google.api.services.vision.v1.model.Feature;

/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with Tensorflow.
 */
public class TensorflowImageListener implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  private static final String MODEL_FILE =
    "file:///android_asset/tensorflow_text_detector.pb";
  private static final String LABEL_FILE =
    "file:///android_asset/text_detector_label_strings.txt";

  private static final int NUM_CLASSES = 2;
  private static final int SEGMENT_SIZE = 128;
  private static final int IMAGE_MEAN = 128;

  private static final float CONF_THRESH = 0.8f;

  // TODO(andrewharp): Get orientation programatically.
  private final int screenRotation = 90;

  private final TensorflowClassifier tensorflow = new TensorflowClassifier();

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;

  private boolean computing = false;
  private Handler handler;

  private RecognitionScoreView scoreView;

  public void initialize(
    final AssetManager assetManager,
    final RecognitionScoreView scoreView,
    final Handler handler) {
    tensorflow.initializeTensorflow(
      assetManager, MODEL_FILE, LABEL_FILE, NUM_CLASSES, SEGMENT_SIZE, IMAGE_MEAN);
    this.scoreView = scoreView;
    this.handler = handler;
  }

  private void drawResizedBitmaps(final Bitmap src, final List<Bitmap> dst) {
    // Resize the original image so that we only get complete segments

    final Matrix matrix = new Matrix();

    final int new_width = (int)(Math.ceil((float)src.getWidth() / SEGMENT_SIZE) * SEGMENT_SIZE);
    final int new_height = (int)(Math.ceil((float)src.getHeight() / SEGMENT_SIZE) * SEGMENT_SIZE);

    final float horizontalScale = new_width / src.getWidth();
    final float verticalScale = new_height / src.getHeight();
    matrix.postScale(horizontalScale, verticalScale);

    LOGGER.v("Image will be scaled to: " + Integer.toString(new_width) + ", " + Integer.toString(new_height));

    final Bitmap resizedBitmap = Bitmap.createBitmap(new_width, new_height, Config.ARGB_8888);
    final Canvas c = new Canvas(resizedBitmap);
    c.drawBitmap(src, matrix, null);

    LOGGER.v("Finished resizing source image.");


    // Extract the segments from the resized image

    for (int horizontal_offset = 0; horizontal_offset < new_width; horizontal_offset += SEGMENT_SIZE)
      for (int vertical_offset = 0; vertical_offset < new_height; vertical_offset += SEGMENT_SIZE) {
      LOGGER.v("Extracting segment: " + Integer.toString(horizontal_offset) + ", " + Integer.toString(vertical_offset));

      dst.add(Bitmap.createBitmap(resizedBitmap, horizontal_offset, vertical_offset, SEGMENT_SIZE, SEGMENT_SIZE));
    }
  }

  // The following is from the sample code for GCV
  // https://github.com/GoogleCloudPlatform/cloud-vision
  private void callCloudVision(final Bitmap bitmap) throws IOException {
    // Switch text to loading
    LOGGER.v("Starting API call");

    // Do the real work in an async task, because we need to use the network anyway
    new AsyncTask<Object, Void, String>() {
      @Override
      protected String doInBackground(Object... params) {
        try {
          HttpTransport httpTransport = AndroidHttp.newCompatibleTransport();
          JsonFactory jsonFactory = GsonFactory.getDefaultInstance();

          Vision.Builder builder = new Vision.Builder(httpTransport, jsonFactory, null);
          builder.setVisionRequestInitializer(new
                                              VisionRequestInitializer(scoreView.getContext().getString(R.string.CloudVisionApiKey)));
          Vision vision = builder.build();

          BatchAnnotateImagesRequest batchAnnotateImagesRequest =
            new BatchAnnotateImagesRequest();
          batchAnnotateImagesRequest.setRequests(new ArrayList<AnnotateImageRequest>() {{
            AnnotateImageRequest annotateImageRequest = new AnnotateImageRequest();

            // Add the image
            com.google.api.services.vision.v1.model.Image base64EncodedImage = new com.google.api.services.vision.v1.model.Image();
            // Convert the bitmap to a JPEG
            // Just in case it's a format that Android understands but Cloud Vision
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, byteArrayOutputStream);
            byte[] imageBytes = byteArrayOutputStream.toByteArray();

            // Base64 encode the JPEG
            base64EncodedImage.encodeContent(imageBytes);
            annotateImageRequest.setImage(base64EncodedImage);

            // add the features we want
            annotateImageRequest.setFeatures(new ArrayList<Feature>() {{
              Feature labelDetection = new Feature();
              labelDetection.setType("TEXT_DETECTION");
              labelDetection.setMaxResults(10);
              add(labelDetection);
            }});

            // Add the list of one thing to the request
            add(annotateImageRequest);
          }});

          Vision.Images.Annotate annotateRequest =
            vision.images().annotate(batchAnnotateImagesRequest);
          // Due to a bug: requests to Vision API containing large images fail when GZipped.
          annotateRequest.setDisableGZipContent(true);
          LOGGER.d("created Cloud Vision request object, sending request");

          BatchAnnotateImagesResponse response = annotateRequest.execute();
          return convertResponseToString(response);

        } catch (GoogleJsonResponseException e) {
          LOGGER.d("failed to make API request because " + e.getContent());
        } catch (IOException e) {
          LOGGER.d("failed to make API request because of other IOException " +
                   e.getMessage());
        }
        return "Cloud Vision API request failed. Check logs for details.";
      }

      protected void onPostExecute(String result) {
        LOGGER.v(result);

        List<Classifier.Recognition> finalResults = new ArrayList<Classifier.Recognition>();
        finalResults.add(new Classifier.Recognition("0", result, 1.0f, null));

        scoreView.setResults(finalResults);
      }
    }.execute();
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;
    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      // No mutex needed as this method is not reentrant.
      if (computing) {
        image.close();
        return;
      }
      computing = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();

      // Initialize the storage bitmaps once when the resolution is known.
      if (previewWidth != image.getWidth() || previewHeight != image.getHeight()) {
        previewWidth = image.getWidth();
        previewHeight = image.getHeight();

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        yuvBytes = new byte[planes.length][];
        for (int i = 0; i < planes.length; ++i) {
          yuvBytes[i] = new byte[planes[i].getBuffer().capacity()];
        }
      }

      for (int i = 0; i < planes.length; ++i) {
        planes[i].getBuffer().get(yuvBytes[i]);
      }

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      ImageUtils.convertYUV420ToARGB8888(
        yuvBytes[0],
        yuvBytes[1],
        yuvBytes[2],
        rgbBytes,
        previewWidth,
        previewHeight,
        yRowStride,
        uvRowStride,
        uvPixelStride,
        false);

      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final List<Bitmap> segments = new ArrayList<Bitmap>();
    drawResizedBitmaps(rgbFrameBitmap, segments);

    handler.post(
      new Runnable() {
        @Override
        public void run() {
          float confText = 0.0f;

          for (final Bitmap segment : segments) {
            final List<Classifier.Recognition> results = tensorflow.recognizeImage(segment);

            if (results.get(0).getTitle().equals("text"))
              confText = Math.max(results.get(0).getConfidence(), confText);
            else
              confText = Math.max(1.0f - results.get(0).getConfidence(), confText);
          }

          if (confText > CONF_THRESH) {
            try {
              callCloudVision(rgbFrameBitmap);
            } catch (IOException exception) {}
          }

          computing = false;
        }
      });

    Trace.endSection();
  }

  private String convertResponseToString(BatchAnnotateImagesResponse response) {
    String message = "";

    List<EntityAnnotation> texts = response.getResponses().get(0).getTextAnnotations();
    if (texts != null) {
      for (EntityAnnotation text : texts) {
        message += String.format("%.3f: %s", text.getScore(), text.getDescription());
        message += "\n";
      }
    } else {
      message += "nothing";
    }

    return message;
  }
}
