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

import junit.framework.Assert;

import com.martinbede.secondsight.env.ImageUtils;
import com.martinbede.secondsight.env.Logger;

import java.util.List;
import java.util.ArrayList;

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
           
           
           List<Classifier.Recognition> finalResults = new ArrayList<Classifier.Recognition>();
          finalResults.add(new Classifier.Recognition("1", "text", confText, null));
          finalResults.add(new Classifier.Recognition("0", "no-text", 1.0f - confText, null));

          scoreView.setResults(finalResults);
           
          computing = false;
         }
       });

    Trace.endSection();
  }
}
