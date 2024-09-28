package ru.khtu.service;

import ai.djl.inference.Predictor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.awt.image.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import ai.djl.*;
import ai.djl.basicmodelzoo.basic.*;
import ai.djl.ndarray.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.translate.*;

@Slf4j
@Service
public class DeepImgClServiceImpl implements DeepImgClService {

    private Image img;
    private Model model;
    private Translator<Image, Classifications> translator;
    private Predictor<Image, Classifications> predictor;
    private Classifications classifications;

    @Override
    public void loadImage() {
        try {
            img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.info("Break when loadImage with Exception : " + e.getClass().getName());
        }
        img.getWrappedImage();
        log.info("Loaded your handwritten digit image");
    }

    @Override
    public void loadModel() {
//        Path modelDir = Paths.get(/*"build/*/"mlp");
        model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
        try {
            model.load(Paths.get("mlp"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.info("Break when loadModel with Exception : " + e.getClass().getName());
        }
        log.info("Model loaded");
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
    public void createPredictor() {
        predictor = model.newPredictor(translator);
        log.info("Predictor created");
    }

    @Override
    public void predictImage() {
        try {
            classifications = predictor.predict(img);
            log.info("Image predicted");
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            log.info("Break when predictImage with Exception : " + e.getClass().getName());
        }
    }

}
