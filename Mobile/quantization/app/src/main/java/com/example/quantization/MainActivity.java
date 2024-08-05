package com.example.quantization;

import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private Spinner modelSpinner, datasetSpinner;
    private TextView baseline, quantization;
    private Button baselineStartbtn, quantizationStartbtn;
    private TextView baselineResult, quantizationResult;
    private ProgressBar progressBar;
    private String selectedModel = null;
    private String selectedDataset = null;
    private String modelPath = "";
    private Interpreter tflite;
    private Bitmap[] images;
    private int[] labels;
    private static final String TAG = "quantization";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setupViews();
        setupSpinners();
    }

    // xml 파일에서 id 가져오는 함수
    private void setupViews() {
        modelSpinner = findViewById(R.id.modelSpinner);
        datasetSpinner = findViewById(R.id.datasetSpinner);
        baseline = findViewById(R.id.baseline);
        quantization = findViewById(R.id.quantization);
        baselineStartbtn = findViewById(R.id.baselineStartbtn);
        quantizationStartbtn = findViewById(R.id.quantizationStartbtn);
        baselineResult = findViewById(R.id.baselineResult);
        quantizationResult = findViewById(R.id.quantizationResult);
        progressBar = findViewById(R.id.midProgressBar);
    }


    private void setupSpinners() {
        ArrayAdapter<CharSequence> modelAdapter = ArrayAdapter.createFromResource(this,
                R.array.models, android.R.layout.simple_spinner_item);
        modelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(modelAdapter);

        ArrayAdapter<CharSequence> datasetAdapter = ArrayAdapter.createFromResource(this,
                R.array.datasets, android.R.layout.simple_spinner_item);
        datasetAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        datasetSpinner.setAdapter(datasetAdapter);

        modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedModel = parent.getItemAtPosition(position).toString();
                updateText();
                loadModelAndData();
                updateModelPath();
                updateButtons();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                selectedModel = null;
            }
        });

        datasetSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedDataset = parent.getItemAtPosition(position).toString();
                updateText();
                loadModelAndData();
                updateButtons();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                selectedDataset = null;
            }
        });
    }
    private void updateModelPath() {
        modelPath = "";
        if (selectedModel != null && selectedDataset != null) {
            modelPath = selectedModel.toLowerCase().replace(" ", "_") + "_" + selectedDataset.toLowerCase() + ".tflite";
        }
    }
    private void updateButtons() {
        String baselineModel = "Baseline/" + modelPath;
        String quantizationModel = "Quantization/qat_" + modelPath;

        baselineStartbtn.setOnClickListener(v -> new LoadModelAndTestTask(baselineModel, baselineResult).execute());
        quantizationStartbtn.setOnClickListener(v -> new LoadModelAndTestTask(quantizationModel, quantizationResult).execute());

        new LoadDataTask().execute();
    }

    private void updateText() {
        if (selectedModel != null && selectedDataset != null) {
            baseline.setText(selectedModel + "\n" + selectedDataset + "\nBaseline");
            quantization.setText(selectedModel + "\n" + selectedDataset + "\nQuantization");
        } else {
            baseline.setText("Please select model and dataset");
            quantization.setText("Please select model and dataset");
        }
    }
    private void loadModelAndData() {
        modelPath = "";
        String imagePath =  "";
        String labelPath = "";

        if (selectedModel != null && selectedDataset != null) {
            if (selectedModel.equals("LeNet 300-100")) {
                modelPath = modelPath + "lenet_";
            }
            else if (selectedModel.equals("AlexNet")) {
                modelPath = modelPath + "alexnet_";
            }
            else if (selectedModel.equals("ResNet18")) {
                modelPath = modelPath + "resnet18_";
            }
            else if (selectedModel.equals("ResNet50")) {
                modelPath = modelPath + "resnet50_";
            }
            else if (selectedModel.equals("MobileNetV1")) {
                modelPath = modelPath + "mobilenetv1_";
            }
            else if (selectedModel.equals("MobileNetV2")) {
                modelPath = modelPath + "mobilenetv2_";
            }
            else {
                modelPath = modelPath + "lenet_";
            }
            if (selectedDataset.equals("MNIST")) {
                modelPath += "mnist.tflite";
                imagePath = "Dataset/" + selectedDataset + "/t10k-images-idx3-ubyte";
                labelPath = "Dataset/" + selectedDataset + "/t10k-labels-idx1-ubyte";
            }
            else if (selectedDataset.equals("Cifar10")) {
                modelPath += "cifar10.tflite";
            }
            Log.v(TAG, imagePath);
            Log.v(TAG, labelPath);
            Log.v(TAG, modelPath);
            new LoadDataTask().execute(imagePath, labelPath);
        }
    }

    private class LoadDataTask extends AsyncTask<String, Void, Boolean> {
        @Override
        protected Boolean doInBackground(String... paths) {
            if (paths.length < 2 || paths[0] == null || paths[1] == null) {
                Log.e(TAG, "Invalid or empty paths provided.");
                return false; // 경로가 유효하지 않을 때 적절한 처리
            }

            try {
                images = loadMNISTImages(paths[0]);
                labels = loadMNISTLabels(paths[1]);
                return true;
            } catch (Exception e) {
                Log.e(TAG, "Failed to load data", e);
                return false;
            }
        }

        @Override
        protected void onPostExecute(Boolean success) {
            if (!success) {
                // 로딩 실패 시 UI 업데이트나 사용자 알림 로직
                Toast.makeText(MainActivity.this, "Error loading data", Toast.LENGTH_SHORT).show();
            }
        }
    }
    private Bitmap[] loadMNISTImages(String fileName) throws IOException {
        InputStream is = getAssets().open(fileName);
        byte[] header = new byte[16];
        is.read(header);

        ByteBuffer bb = ByteBuffer.wrap(header);
        int magic = bb.getInt();
        int numImages = bb.getInt();
        int numRows = bb.getInt();
        int numCols = bb.getInt();

        Bitmap[] images = new Bitmap[numImages];
        byte[] imageBytes = new byte[numRows * numCols];

        for (int i = 0; i < numImages; i++) {
            is.read(imageBytes);
            images[i] = convertByteArrayToBitmap(imageBytes, numRows, numCols);
        }
        is.close();
        Log.d("완료", "이미지 로드 완료");
        return images;
    }
    private int[] loadMNISTLabels(String fileName) throws IOException {
        InputStream is = getAssets().open(fileName);
        byte[] header = new byte[8];
        is.read(header, 0, 8);

        ByteBuffer bb = ByteBuffer.wrap(header);
        int magic = bb.getInt();
        int numLabels = bb.getInt();

        int[] labels = new int[numLabels];
        for (int i = 0; i < numLabels; i++) {
            labels[i] = is.read(); // 이 부분은 unsigned byte를 읽어야 하므로, 0xFF와 AND 연산을 수행할 수도 있습니다.
        }
        is.close();
        Log.d("완료", "라벨 로드 완료");
        return labels;
    }
    private ByteBuffer loadModelFile(String modelName) throws IOException {
        // Android의 AssetManager를 사용하여 modelName 파라미터로 지정된 파일 이름을 가진 asset을 열어 InputStream을 생성한다.
        InputStream inputStream = getAssets().open(modelName);
        // 스트림에서 읽을 수 있는 바이트 수를 반환한다. 이 메소드는 파일의 전체 크기를 알아내는 데 사용된다.
        byte[] modelBytes = new byte[inputStream.available()];
        // 배열의 크기만큼 파일에서 데이터를 읽어 배열에 저장한다. 이 배열은 모델 파일의 전체 내용을 바이트로 포함하게 된다.
        inputStream.read(modelBytes);
        // 저장된 크기의 직접 버퍼를 생성한다. 직접 버퍼는 메모리를 효율적으로 사용하며, 파일의 I/O와 네이티브 코드에서의 데이터 전송 시 성능이 빠르다.
        ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelBytes.length);
        // modelBytes에 저장된 모델 데이터를 ByteBuffer에 복사한다.
        modelBuffer.put(modelBytes);
        // 버퍼의 위치를 0으로 설정한다. 버퍼의 데이터를 처음부터 읽거나 쓸 수 있게 준비하는 과정이다.
        modelBuffer.rewind();
        return modelBuffer;
    }
    private class LoadModelAndTestTask extends AsyncTask<Void, Integer, String> {
        private String modelName;
        private TextView targetTextView;
        LoadModelAndTestTask(String modelName, TextView targetTextView) {
            this.modelName = modelName;
            this.targetTextView = targetTextView;
        }
        @Override
        protected void onPreExecute() {
            progressBar.setVisibility(View.VISIBLE);
            progressBar.setMax(100);
        }

        @Override
        protected String doInBackground(Void... voids) {
            // 이미지와 레이블이 로드되었는지 확인
            if (images == null || labels == null) {
                return "Data not loaded properly.";
            }

            try {
                tflite = new Interpreter(loadModelFile(modelName));
                Log.v(TAG, "Model loaded successfully.");
            } catch (IOException e) {
                Log.e(TAG, "Failed to load model.", e);
                cancel(false);
                return "Model load failed.";
            }

            warmUpModel(); // 예열
            float totalInferenceTime = measureInferenceTime(); // 인퍼런스 시간 측정
            float accuracy = calculateAccuracy(); // 정확도 계산
            return String.format("Accuracy: %.2f%%,\nAverage Inference Time: %.2f ms", accuracy, totalInferenceTime);
        }
        private int runInference(Bitmap image) {
            ByteBuffer input = convertBitmapToByteBuffer(image);
            float[][] output = new float[1][10];
            tflite.run(input, output);
            return argMax(output[0]);
        }
        private void warmUpModel() {
            for (int i = 0; i < 10; i++) {
                runInference(images[i % images.length]);
            }
        }
        private float measureInferenceTime() {
            float totalInferenceTime = 0;
            int numberOfRuns = 100;
            for (int i = 0; i < numberOfRuns; i++) {
                float startTime = SystemClock.elapsedRealtime();
                runInference(images[i % images.length]);
                float endTime = SystemClock.elapsedRealtime();
                totalInferenceTime += (endTime - startTime);
                publishProgress((i + 1) * 100 / numberOfRuns);
                SystemClock.sleep(10);
            }
            return totalInferenceTime / numberOfRuns;
        }
        private float calculateAccuracy() {
            int correctPredictions = 0;
            for (int i = 0; i < images.length; i++) {
                int predictedLabel = runInference(images[i]);
                if (predictedLabel == labels[i]) {
                    correctPredictions++;
                }
            }
            return (correctPredictions / (float) images.length) * 100;
        }
        @Override
        protected void onProgressUpdate(Integer... values) {
            progressBar.setProgress(values[0]);
        }

        @Override
        protected void onPostExecute(String result) {
            progressBar.setVisibility(View.INVISIBLE);
            targetTextView.setText(result);
        }
    }
    private Bitmap convertByteArrayToBitmap(byte[] imageBytes, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = 0xFF & imageBytes[y * width + x];
                int value = 0xFF000000 | (pixel << 16) | (pixel << 8) | pixel;
                bitmap.setPixel(x, y, value);
            }
        }
        return bitmap;
    }
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        /*
        resolution이 28인 경우
         */
//        int bytesPerPixel = 4; // Float 사용 시 4바이트 필요
//        // 지정된 크기의 직접 버퍼를 생성한다. 직접 버퍼는 Java 가상 머신(JVM)의 힙 외부에 메모리를 할당하여, 파일 I/O와 같이 네이티브 I/O 작업을 더 효율적으로 수행할 수 있게 한다.
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bitmap.getHeight() * bitmap.getWidth() * bytesPerPixel);
//        // 시스템의 네이티브 바이트 순서를 버퍼의 바이트 순서로 설정합니다.
//        byteBuffer.order(ByteOrder.nativeOrder());
//        // 모든 픽셀의 색상 데이터를 추출하여 정수 배열 pixels에 저장한다.
//        int[] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
//        //  각 정수는 픽셀의 색상 값을 ARGB 형식으로 포함하고 있다.
//        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//        // 반복문은 각 픽셀에 대해 실행된다. MNIST 데이터는 흑백 이미지이므로 하나의 색상 채널만 필요, 그 값을 255.0으로 나누어 0과 1 사이의 범위로 정규화한다.
//        for (int pixel : pixels) {
//            // MNIST는 흑백 이미지이므로, RGB 중 하나만 읽어도 충분합니다. 여기서는 빨간색 채널을 사용합니다.
//            int pixelValue = (pixel >> 16) & 0xFF;
//            byteBuffer.putFloat(pixelValue / 255.0f);  // 모델 입력을 위해 0과 1 사이의 값으로 정규화
//        }
//        return byteBuffer;

        /*
        resolution이 48인 경우
         */
//        int DIM_IMG_SIZE_X = 48; // 모델이 요구하는 이미지 너비
//        int DIM_IMG_SIZE_Y = 48; // 모델이 요구하는 이미지 높이
//        int DIM_PIXEL_SIZE = 3;   // 모델이 요구하는 채널 수 (RGB)
//
//        bitmap = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, false);
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//
//        int[] pixels = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
//        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
//        for (int pixel : pixels) {
//            byteBuffer.putFloat((pixel >> 16) & 0xFF);
//            byteBuffer.putFloat((pixel >> 8) & 0xFF);
//            byteBuffer.putFloat(pixel & 0xFF);
//        }
//        return byteBuffer;

        int DIM_IMG_SIZE_X = 32; // 모델이 요구하는 이미지 너비
        int DIM_IMG_SIZE_Y = 32; // 모델이 요구하는 이미지 높이
        int DIM_PIXEL_SIZE = 3;   // 모델이 요구하는 채널 수 (RGB)

        bitmap = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, false);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
        for (int pixel : pixels) {
            byteBuffer.putFloat((pixel >> 16) & 0xFF);
            byteBuffer.putFloat((pixel >> 8) & 0xFF);
            byteBuffer.putFloat(pixel & 0xFF);
        }
        return byteBuffer;
    }
    private int argMax(float[] floats) {
        int maxIndex = -1;
        float max = 0.0f;
        for (int i = 0; i < floats.length; i++) {
            if (floats[i] > max) {
                max = floats[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
