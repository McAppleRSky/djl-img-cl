package ru.khtu.component;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;
import ru.khtu.service.DeepImgClService;
import ru.khtu.service.ImgClService;

import javax.imageio.ImageIO;
import java.awt.*;
import java.net.URL;

@Slf4j
@Component
@RequiredArgsConstructor
public class Runner implements CommandLineRunner {

//    private final DeepImgClService deepImgClService;
    private final ImgClService imgClService;

    @Override
    public void run(String... args) throws Exception {
        log.info("Run CommandLine Runner Component ...");
        try {
            imgClService.retrieveTrainedModel();
            imgClService.createTranslator();
            imgClService.retrieveHandwrittenImage();
            imgClService.predictImage();
            log.info("CommandLine Runner running complete");
        } catch (Exception e) {
            log.warn("Brake with Exception");
        }
        java.awt.image.BufferedImage trainingDataSetDemo = null;
        try {
            URL url = new URL("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png");
            trainingDataSetDemo = ImageIO.read(url);
        } catch (Exception e) {
            log.warn("Brake demonstrate Train DataSet with exception : " + e.getClass().getName());
        }
    }

}
