package ru.khtu.service;

public interface DeepImgClService {

    void loadImage();

    void loadModel();

    void createTranslator();

    void createPredictor();

    void predictImage();

}
