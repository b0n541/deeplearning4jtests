package net.b0n541.deeplearning4j;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class SimpleXORTest {

	public static void main(String[] args) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.iterations(1)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(0.05)
				.miniBatch(false)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(2).nOut(4).activation(Activation.SIGMOID).build())
				.layer(1, new OutputLayer.Builder().nIn(4).nOut(2).activation(Activation.SOFTMAX).build())
				.backprop(true)
				.build();
		
		 MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // add an listener which outputs the error every 100 parameter updates
        net.setListeners(new ScoreIterationListener(100));

        printLayers(net);
        
        for (int i = 0; i < 10000; i++) {
        	net.fit(getTrainingData());
        	if (net.gradientAndScore().getValue() < 0.01) {
        		System.out.println("Score: "+net.gradientAndScore().getValue());
        		System.out.println("after "+i +" iterations...");
        		break;
        	}
        }

        	INDArray inputs = getTrainingData().getFeatureMatrix();
        	System.out.println(inputs);
       INDArray outputs = net.output(inputs);
       System.out.println(outputs);
       
       Evaluation eval = new Evaluation(2);
       eval.eval(getTrainingData().getLabels(), outputs);
       System.out.println(eval.stats());
	}

	private static void printLayers(MultiLayerNetwork net) {
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);
	}
	
	private static DataSet getTrainingData() {
	       // list off input values, 4 training samples with data for 2
        // input-neurons each
        INDArray input = Nd4j.zeros(4, 2);

        // correspondending list with expected output values, 4 training samples
        // with data for 2 output-neurons each
        INDArray outputs = Nd4j.zeros(4, 2);

        // create first dataset
        // when first input=0 and second input=0
        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        // then the first output fires for false, and the second is 0 (see class
        // comment)
        outputs.putScalar(new int[]{0, 0}, 1);
        outputs.putScalar(new int[]{0, 1}, 0);

        // when first input=1 and second input=0
        input.putScalar(new int[]{1, 0}, 1);
        input.putScalar(new int[]{1, 1}, 0);
        // then xor is true, therefore the second output neuron fires
        outputs.putScalar(new int[]{1, 0}, 0);
        outputs.putScalar(new int[]{1, 1}, 1);

        // same as above
        input.putScalar(new int[]{2, 0}, 0);
        input.putScalar(new int[]{2, 1}, 1);
        outputs.putScalar(new int[]{2, 0}, 0);
        outputs.putScalar(new int[]{2, 1}, 1);

        // when both inputs fire, xor is false again - the first output should
        // fire
        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);
        outputs.putScalar(new int[]{3, 0}, 1);
        outputs.putScalar(new int[]{3, 1}, 0);

        // create dataset object
       return new DataSet(input, outputs);

	}
}
