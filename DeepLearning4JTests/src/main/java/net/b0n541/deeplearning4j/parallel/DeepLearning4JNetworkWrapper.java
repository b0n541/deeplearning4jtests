/**
 * Copyright (C) 2017 Jan Schäfer (jansch@users.sourceforge.net)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.b0n541.deeplearning4j.parallel;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Wraps the DeepLearning4J network to fulfill the interface
 * {@link NeuralNetwork}
 */
public class DeepLearning4JNetworkWrapper {

	private final MultiLayerNetwork net;
	private final NetworkTopology topo;

	private static final double LOWER_DIST_BOUND = -0.1;
	private static final double UPPER_DIST_BOUND = 0.1;

	/**
	 * Constructor
	 *
	 * @param topo
	 *            Network topology
	 * @param useBias
	 *            TRUE, if bias nodes should be used
	 */
	public DeepLearning4JNetworkWrapper(final NetworkTopology topo, final boolean useBias) {

		this.topo = topo;

		// Set up network configuration
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		// how often should the training set be run, we need something above
		// 1000, or a higher learning-rate - found this values just by trial and
		// error
		builder.iterations(1);
		// learning rate
		builder.learningRate(0.1);
		// fixed seed for the random generator, so any run of this program
		// brings the same results - may not work if you do something like
		// ds.shuffle()
		builder.seed(123);
		// not applicable, this network is to small - but for bigger networks it
		// can help that the network will not only recite the training data
		builder.useDropConnect(false);
		// a standard algorithm for moving on the error-plane, this one works
		// best for me, LINE_GRADIENT_DESCENT or CONJUGATE_GRADIENT can do the
		// job, too - it's an empirical value which one matches best to
		// your problem
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		// init the bias with 0 - empirical value, too
		builder.biasInit(0);
		// from "http://deeplearning4j.org/architecture": The networks can
		// process the input more quickly and more accurately by ingesting
		// minibatches 5-10 elements at a time in parallel.
		// this example runs better without, because the dataset is smaller than
		// the mini batch size
		builder.miniBatch(false);

		// create a multilayer network with 2 layers (including the output
		// layer, excluding the input payer)
		ListBuilder listBuilder = builder.list();

		DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
		// two input connections - simultaneously defines the number of input
		// neurons, because it's the first non-input-layer
		hiddenLayerBuilder.nIn(2);
		// number of outgooing connections, nOut simultaneously defines the
		// number of neurons in this layer
		hiddenLayerBuilder.nOut(4);
		// put the output through the sigmoid function, to cap the output
		// valuebetween 0 and 1
		hiddenLayerBuilder.activation(Activation.SIGMOID);
		// random initialize weights with values between 0 and 1
		hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

		// build and set as layer 0
		listBuilder.layer(0, hiddenLayerBuilder.build());

		// MCXENT or NEGATIVELOGLIKELIHOOD (both are mathematically equivalent) work ok
		// for this example - this
		// function calculates the error-value (aka 'cost' or 'loss function value'),
		// and quantifies the goodness
		// or badness of a prediction, in a differentiable way
		// For classification (with mutually exclusive classes, like here), use
		// multiclass cross entropy, in conjunction
		// with softmax activation function
		Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
		// must be the same amout as neurons in the layer before
		outputLayerBuilder.nIn(4);
		// two neurons in this layer
		outputLayerBuilder.nOut(2);
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		outputLayerBuilder.dist(new UniformDistribution(0, 1));
		listBuilder.layer(1, outputLayerBuilder.build());

		// no pretrain phase for this network
		listBuilder.pretrain(false);

		// seems to be mandatory
		// according to agibsonccc: You typically only use that with
		// pretrain(true) when you want to do pretrain/finetune without changing
		// the previous layers finetuned weights that's for autoencoders and
		// rbms
		listBuilder.backprop(true);

		// build and init the network, will check if everything is configured
		// correct
		MultiLayerConfiguration conf = listBuilder.build();
		net = new MultiLayerNetwork(conf);
		net.init();
	}

	public double adjustWeights(final double[] inputs, final double[] outputs) {

		final INDArray input = Nd4j.zeros(1, inputs.length);
		final INDArray output = Nd4j.zeros(1, outputs.length);

		for (int i = 0; i < inputs.length; i++) {
			input.putScalar(new int[] { 0, i }, inputs[i]);
		}
		for (int i = 0; i < outputs.length; i++) {
			output.putScalar(new int[] { 0, i }, outputs[i]);
		}

		net.fit(new DataSet(input, output));

		return net.gradientAndScore().getValue();
	}

	public double adjustWeightsBatch(final double[][] inputs, final double[][] outputs) {

		final INDArray input = Nd4j.zeros(inputs.length, inputs[0].length);
		final INDArray output = Nd4j.zeros(outputs.length, outputs[0].length);

		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				input.putScalar(new int[] { i, j }, inputs[i][j]);
			}
		}

		for (int i = 0; i < outputs.length; i++) {
			for (int j = 0; j < outputs[i].length; j++) {
				output.putScalar(new int[] { i, j }, outputs[i][j]);
			}
		}

		net.fit(new DataSet(input, output));

		return net.gradientAndScore().getValue();
	}

	public void resetNetwork() {
		net.init();
	}

	public double[] getPredictedOutcome(final double[] inputs) {
		final INDArray output = net.output(Nd4j.create(inputs));
		final double[] result = new double[output.length()];
		for (int i = 0; i < output.length(); i++) {
			result[i] = output.getDouble(i);
		}
		return result;
	}

	public long getIterations() {
		return net.getLayerWiseConfigurations().getIterationCount();
	}

	public boolean saveNetwork(final String fileName) {
		// TODO Auto-generated method stub
		return false;
	}

	public void loadNetwork(final String fileName) {
		// TODO Auto-generated method stub
	}
}
