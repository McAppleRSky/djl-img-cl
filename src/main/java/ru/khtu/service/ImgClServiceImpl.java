package ru.khtu.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Slf4j
@Service
public class ImgClServiceImpl implements ImgClService {

//    final Logger logger = LoggerFactory.getLogger(this.getClass());

    private final int INPUT_SIZE = 28 * 28, // 784
            OUTPUT_SIZE = 10,
            EPOCH_COUNT = 2;
    private final String LOCAL_PATH = "build/mlp";
    private Image image;
    private Model model;
    private Translator<Image, Classifications> translator;
    private Classifications classifications;

    @Value("${model}")
    private String modelArgCondition;

    @PostConstruct
    public void createTrainingAndSaveModel() {
        if ("# needTrainingAndSave".equals(modelArgCondition)) {
            Model modelLocal = createModel();
            Mnist dataSet = initTrainDataSet();
            trainModel(modelLocal, dataSet);
            saveModel(modelLocal);
        } else {
            log.info("Bypass creating, training and save Model");
        }
    }

    private Model createModel() {
        Model result = Model.newInstance("mlp");
        result.setBlock(new Mlp(INPUT_SIZE, OUTPUT_SIZE, new int[] {128, 64}));
        log.info("Model created and to be training");
        return result;
    }

    private Mnist initTrainDataSet() {
        final int BATCH_SIZE = 32;
        Mnist basicDataSet = Mnist.builder().setSampling(BATCH_SIZE, true).build();
        try {
            basicDataSet.prepare(new ProgressBar());
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.warn("Brake initTrainDataSet with exception : " + e.getClass().getName());
        }
        log.info("DataSet initiated");
        return basicDataSet;
    }

    private void trainModel(Model modelLocal, Mnist dataSet) {
        DefaultTrainingConfig trainingConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                //softmaxCrossEntropyLoss is a standard loss for classification problems
                .addEvaluator(new Accuracy()) // Use accuracy so we humans can understand how accurate the model is
                .addTrainingListeners(TrainingListener.Defaults.logging());
        // Now that we have our training configuration, we should create a new trainer for our model
        Trainer trainer = modelLocal.newTrainer(trainingConfig);
        trainer.initialize(new Shape(1, INPUT_SIZE));
        // Deep learning is typically trained in epochs where each epoch trains the model on each item in the dataset once.
        try {
            EasyTrain.fit(trainer, EPOCH_COUNT, dataSet, null);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.warn("Brake trainModel with exception : " + e.getClass().getName());
        }
        log.info("Model trained");
    }

    private void saveModel(Model modelLocal) {
        Path modelDir = Paths.get(LOCAL_PATH);
        try {
            Files.createDirectories(modelDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.warn("Brake mkdir (PostConstruct) with exception : " + e.getClass().getName());
        }
        modelLocal.setProperty("Epoch", String.valueOf(EPOCH_COUNT));
        try {
            modelLocal.save(modelDir, "mlp");
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.warn("Brake saving with exception : " + e.getClass().getName());
        }
        log.info("Model saved. Summary:\n" + modelLocal);
    }

    @Override
    public void retrieveTrainedModel() {
        model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
        try {
            model.load(Paths.get(LOCAL_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void createTranslator() {
        translator = new Translator<>() {

            @Override
            public NDList processInput(TranslatorContext context, Image input) throws Exception {
                NDArray array = input.toNDArray(context.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext translatorContext, NDList list) throws Exception {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream
                        .range(0, 10)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }

        };
        log.info("Translator created");
    }

    @Override
    public void retrieveHandwrittenImage() {
        try {
            image = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.warn("Brake downloadImage with exception : " + e.getClass().getName());
        }
        image.getWrappedImage();
        log.info("Image downloaded");
    }

    @Override
    public void predictImage() {
        Predictor<Image, Classifications> predictor = model.newPredictor(translator);
        try {
            classifications = predictor.predict(image);
            log.info("Image predicted");
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.info("Break when predictImage with Exception : " + e.getClass().getName());
        }
        Pair<Double, Integer[]> pairRelevanceValues = maxClass(classifications.getProbabilities());
        switch (pairRelevanceValues.getValue().length) {
            case 1 :
                log.info("Image classified as : " + pairRelevanceValues.getValue()[0] + "\n" + classifications);
                break;
            case 0 :
                log.warn("Image not classified" + pairRelevanceValues.getValue()[0] + "\n" + classifications);
                break;
            default:
                log.warn(
                        "Image classified many digit with Relevance" + pairRelevanceValues.getKey() + "\n"
                                + classifications );
        }
    }

    Pair<Double, Integer[]> maxClass(List<Double> classifications) {
        Double relevanceMax = Collections.max(classifications);
        List<Integer> values = new ArrayList();
        for (int i = 0; i < classifications.size(); i++) {
            if (relevanceMax.equals(classifications.get(i))) {
                values.add(i);
            }
        }
        return new Pair(relevanceMax, values.toArray(new Integer[values.size()]));
    }

}
