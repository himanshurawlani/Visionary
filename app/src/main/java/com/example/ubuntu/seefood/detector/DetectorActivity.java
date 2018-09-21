/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.ubuntu.seefood.detector;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Toast;

import com.example.ubuntu.seefood.R;
import com.example.ubuntu.seefood.detector.OverlayView.DrawCallback;
import com.example.ubuntu.seefood.env.BorderedText;
import com.example.ubuntu.seefood.env.ImageUtils;
import com.example.ubuntu.seefood.env.Logger;
import com.example.ubuntu.seefood.tracking.MultiBoxTracker;
import com.microsoft.projectoxford.face.FaceServiceClient;
import com.microsoft.projectoxford.face.contract.Face;
import com.microsoft.projectoxford.face.contract.FaceAttribute;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.Vector;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    private final int RESULT_OK = 2;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;
    private Classifier detector;
    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private boolean computingDetection = false;
    private long timestamp = 0;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private byte[] luminanceCopy;
    private BorderedText borderedText;
    private HashSet<Classifier.Recognition> resultSet = new HashSet<>();

    public boolean apiCallActive = false;

    // TODO: Check what params are received in 'objects'
    @SuppressLint("StaticFieldLeak")
    AsyncTask<assistantParams, Void, Void> backgroundVoiceAssistant = new
            AsyncTask<assistantParams, Void, Void>() {
                @Override
                protected Void doInBackground(assistantParams... object) {
                    voiceAssistant(object[0].objectCount, object[0].croppedBitmap);
                    return null;
                }
            };

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        final HashMap<String, Integer> objectCount = new HashMap<>();
                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {

                                // ResultSet is a hashset which stores all the recognitions while the camera is rolled on objects
                                // This resultSet is passed to the DisplayResults activity
                                // ResultSet stores objects of class Classifier.Recognition - which has many fields like id, name etc
                                resultSet.add(result);
                                LOGGER.d("Detection Result: " + result.toString());
                                String className = result.getTitle();
                                if (objectCount.containsKey(className)) {
                                    objectCount.put(className, objectCount.get(className) + 1);
                                } else {
                                    objectCount.put(className, 1);
                                }

                                canvas.drawRect(location, paint);
                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        LOGGER.d("AsyncTask.Status: " + backgroundVoiceAssistant.getStatus());
                        // Creating a background thread for TextToSpeech
                        if (!tts.isSpeaking() &&
                                backgroundVoiceAssistant.getStatus() != AsyncTask.Status.RUNNING) {

                            backgroundVoiceAssistant = new AsyncTask<assistantParams, Void, Void>() {
                                @Override
                                protected Void doInBackground(assistantParams... object) {
                                    voiceAssistant(object[0].objectCount, object[0].croppedBitmap);
                                    return null;
                                }
                            };
                            backgroundVoiceAssistant.execute(
                                    new assistantParams(objectCount, croppedBitmap));
                        }

//                        if (!tts.isSpeaking()) {
//                            AsyncTask.execute(new Runnable() {
//                                @Override
//                                public void run() {
//                                    voiceAssistant(objectCount, croppedBitmap);
//                                }
//                            });
//                        }

                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        trackingOverlay.postInvalidate();

                        requestRender();
                        computingDetection = false;
                    }
                });
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }


        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        if (!isDebug()) {
                            return;
                        }
                        final Bitmap copy = cropCopyBitmap;
                        if (copy == null) {
                            return;
                        }

                        final int backgroundColor = Color.argb(100, 0, 0, 0);
                        canvas.drawColor(backgroundColor);

                        final Matrix matrix = new Matrix();
                        final float scaleFactor = 2;
                        matrix.postScale(scaleFactor, scaleFactor);
                        matrix.postTranslate(
                                canvas.getWidth() - copy.getWidth() * scaleFactor,
                                canvas.getHeight() - copy.getHeight() * scaleFactor);
                        canvas.drawBitmap(copy, matrix, new Paint());

                        final Vector<String> lines = new Vector<String>();
                        if (detector != null) {
                            final String statString = detector.getStatString();
                            final String[] statLines = statString.split("\n");
                            for (final String line : statLines) {
                                lines.add(line);
                            }
                        }
                        lines.add("");

                        lines.add("Frame: " + previewWidth + "x" + previewHeight);
                        lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                        lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                        lines.add("Rotation: " + sensorOrientation);
                        lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                    }
                });
    }

    public void voiceAssistant(HashMap<String, Integer> objectCount, final Bitmap croppedBitmap) {

        StringBuilder speak = new StringBuilder();
        if (objectCount.size() > 0) {
            speak.append("Found ");
            for (Map.Entry<String, Integer> cls : objectCount.entrySet()) {
                speak.append(cls.getValue()).append(" ").append(cls.getKey()).append(", ");
            }

            // Creating a background thread for face detection
            if (objectCount.containsKey("person") && objectCount.get("person") > 0 && !apiCallActive
                    && !tts.isSpeaking() && isOnline()) {
                AsyncTask.execute(new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.d("MICROSOFT API: detectFaces called!");
                        detectFaces(croppedBitmap);
                    }
                });
            }

//            // ML kit OCR
//            AsyncTask.execute((new Runnable() {
//                @Override
//                public void run() {
//                    FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(croppedBitmap);
//                    FirebaseVisionTextRecognizer detector = FirebaseVision.getInstance()
//                            .getOnDeviceTextRecognizer();
//                    detector.processImage(image)
//                            .addOnSuccessListener(
//                                    new OnSuccessListener<FirebaseVisionText>() {
//                                        @Override
//                                        public void onSuccess(FirebaseVisionText texts) {
//                                            processTextRecognitionResult(texts);
//                                        }
//                                    })
//                            .addOnFailureListener(
//                                    new OnFailureListener() {
//                                        @Override
//                                        public void onFailure(@NonNull Exception e) {
//                                            // Task failed with an exception
//                                            e.printStackTrace();
//                                        }
//                                    });
//                }
//            }));
            tts.speak(speak.toString(), TextToSpeech.QUEUE_FLUSH, null);
            LOGGER.d("SPEECH: " + speak.toString());
        }
    }

    public boolean isOnline() {
        ConnectivityManager cm =
                (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo netInfo = cm.getActiveNetworkInfo();
        return netInfo != null && netInfo.isConnectedOrConnecting();
    }

    public class assistantParams {
        HashMap<String, Integer> objectCount;
        Bitmap croppedBitmap;

        assistantParams(HashMap<String, Integer> objectCount, Bitmap croppedBitmap) {
            this.objectCount = objectCount;
            this.croppedBitmap = croppedBitmap;
        }
    }

    public void detectFaces(Bitmap imageBitmap) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        ByteArrayInputStream inputStream =
                new ByteArrayInputStream(outputStream.toByteArray());

        // TODO: Check for memory leaks here!
        @SuppressLint("StaticFieldLeak") AsyncTask<InputStream, String, Face[]> detectTask =
                new AsyncTask<InputStream, String, Face[]>() {
                    String exceptionMessage = "";

                    @Override
                    protected Face[] doInBackground(InputStream... params) {
                        try {
                            LOGGER.d("MICROSOFT API: Detecting...");
                            Face[] result = faceServiceClient.detect(
                                    params[0],
                                    true,         // returnFaceId
                                    false,        // returnFaceLandmarks
                                    new FaceServiceClient.FaceAttributeType[]{
                                            FaceServiceClient.FaceAttributeType.Age,
                                            FaceServiceClient.FaceAttributeType.Gender,
                                            FaceServiceClient.FaceAttributeType.Emotion} // return FaceAttributes:
                            );
                            if (result == null) {
                                LOGGER.d("MICROSOFT API: Detection Finished. Nothing detected");
                                return null;
                            }
                            LOGGER.d("MICROSOFT API: " +
                                    String.format("Detection Finished. %d face(s) detected",
                                            result.length));
                            return result;
                        } catch (Exception e) {
                            exceptionMessage = String.format(
                                    "Detection failed: %s", e.getMessage());
                            return null;
                        }
                    }

                    @Override
                    protected void onPreExecute() {
                        LOGGER.d("MICROSOFT API: onPreExecute() Called!");
                        apiCallActive = true;
                    }

                    @Override
                    protected void onProgressUpdate(String... progress) {
                        LOGGER.d("MICROSOFT API: onProgressUpdate() called");
                    }

                    @Override
                    protected void onPostExecute(Face[] result) {
                        if (!exceptionMessage.equals("")) {
                            showError(exceptionMessage);
                        }
                        if (result == null) return;
                        StringBuilder speak = new StringBuilder();
                        int count = 1;
                        for (Face face : result) {
                            UUID id = face.faceId;
                            FaceAttribute fa = face.faceAttributes;
                            LOGGER.d("MICROSOFT API: ID: " + id + " FA: " + fa);
                            speak.append("Person ").append(count++).append(" is ").append((int) fa.age)
                                    .append(" years old, ").append(fa.gender).append(". ");
                            LOGGER.d("MICROSOFT API: FA(Emotion): Anger=" + fa.emotion.anger +
                                    " \nContempt=" + fa.emotion.contempt +
                                    " \nDisgust=" + fa.emotion.disgust +
                                    " \nFear=" + fa.emotion.fear +
                                    " \nHappiness=" + fa.emotion.happiness +
                                    " \nNeutral=" + fa.emotion.neutral +
                                    " \nSadness=" + fa.emotion.sadness +
                                    " \nSurprise" + fa.emotion.surprise);

                            if (fa.gender.equals("male")) {
                                speak.append("He looks ");
                            } else {
                                speak.append("She looks ");
                            }

                            if (fa.emotion.anger > 0) {
                                speak.append("angry. ");
                            } else if (fa.emotion.contempt > 0) {
                                speak.append("contempt. ");
                            } else if (fa.emotion.disgust > 0) {
                                speak.append("disgusted. ");
                            } else if (fa.emotion.fear > 0) {
                                speak.append("scared. ");
                            } else if (fa.emotion.surprise > 0) {
                                speak.append("surprised. ");
                            } else if (fa.emotion.neutral > 0) {
                                speak.append("neutral. ");
                            } else if (fa.emotion.sadness > 0) {
                                speak.append("contempt. ");
                            } else {
                                speak.append("happy. ");
                            }
                        }
                        LOGGER.d("MICROSOFT API SPEECH: " + speak);
                        tts.speak(speak.toString(), TextToSpeech.QUEUE_ADD, null);
                        apiCallActive = false;
                    }
                };

        detectTask.execute(inputStream);
    }

    private void showError(String message) {
        new AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage(message)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                    }
                })
                .create().show();
    }

//    private void processTextRecognitionResult(FirebaseVisionText texts) {
//        List<FirebaseVisionText.TextBlock> blocks = texts.getTextBlocks();
//        if (blocks.size() == 0) {
//            return;
//        }
//        StringBuilder speak = new StringBuilder();
//        for (int i = 0; i < blocks.size(); i++) {
//            List<FirebaseVisionText.Line> lines = blocks.get(i).getLines();
//            for (int j = 0; j < lines.size(); j++) {
//                List<FirebaseVisionText.Element> elements = lines.get(j).getElements();
//                for (int k = 0; k < elements.size(); k++) {
//                    speak.append(elements.get(k).getText()).append(" ");
//                }
//            }
//        }
//        LOGGER.d("MLKIT OCR: " + speak.toString());
//    }

    @Override
    public synchronized void onPause() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onPause();
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onSetDebug(final boolean debug) {
        detector.enableStatLogging(debug);
    }

    /******************** Code to return detected objects to the CameraActivity.java ******************/

    public void closeCameraActivity(View view) {
        Intent intent = getIntent();
        Bundle resultsBundle = new Bundle();

        int counter = 1;
        for (Classifier.Recognition recognition : resultSet) {
            resultsBundle.putString("Object" + counter++,
                    "Class:" + recognition.getTitle() + ", Confidence:" + recognition.getConfidence() * 100);
        }
        intent.putExtra("resultsBundle", resultsBundle);
        setResult(RESULT_OK, intent);

        finish();
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API
    }
}
