
Java
package com.example.ia;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;

import java.io.File;

@Service
public class AIService {

    private final MultiLayerNetwork model;
    private final WordVectors wordVectors;

    public AIService() {
        try {
            // Cargar modelo neuronal entrenado
            File modelFile = new File("src/main/resources/modelo.zip");
            if (!modelFile.exists()) {
                throw new RuntimeException("No se encontró modelo.zip en resources");
            }
            model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            // Cargar embeddings Word2Vec
            File vecFile = new File("src/main/resources/word2vec.txt");
            if (!vecFile.exists()) {
                throw new RuntimeException("No se encontró word2vec.txt en resources");
            }
            wordVectors = WordVectorSerializer.readWord2VecModel(vecFile);

        } catch (Exception e) {
            throw new RuntimeException("Error cargando IA: " + e.getMessage(), e);
        }
    }

    public String clasificarTexto(String texto) {
        try {
            INDArray features = fraseAEmbedding(texto);
            INDArray output = model.output(features);
            int clase = output.argMax(1).getInt(0);

            return switch (clase) {
                case 0 -> "Positivo";
                case 1 -> "Negativo";
                case 2 -> "Neutral";
                default -> "Desconocido";
            };
        } catch (Exception e) {
            return "Error procesando texto: " + e.getMessage();
        }
    }

    private INDArray fraseAEmbedding(String frase) {
        String[] tokens = StringUtils.split(frase.toLowerCase());
        INDArray sum = org.nd4j.linalg.factory.Nd4j.zeros(wordVectors.lookupTable().layerSize());
        int count = 0;
        for (String token : tokens) {
            if (wordVectors.hasWord(token)) {
                sum.addi(wordVectors.getWordVectorMatrix(token));
                count++;
            }
        }
        return count > 0 ? sum.div(count) : sum;
    }
}


