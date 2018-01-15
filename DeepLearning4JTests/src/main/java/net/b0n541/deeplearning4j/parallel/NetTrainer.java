package net.b0n541.deeplearning4j.parallel;

import java.util.Random;

public class NetTrainer {

	Random random = new Random();
	private String name;
	private DeepLearning4JNetworkWrapper wrapper;
	private long maxIterations = 10L;

	public NetTrainer(String name, long maxIterations) {
		this.name = name;
		this.wrapper = new DeepLearning4JNetworkWrapper(new NetworkTopology(2, new int[] { 3 }, 1), false);
		this.maxIterations = maxIterations;
	}

	public void trainNet() {
		System.out.println(name + " on thread " + Thread.currentThread().getName() + " started...");
		do {
			double[] inputs = random.doubles(2, 0.0, 1.0).toArray();
			double[] predictedOutcome = wrapper.getPredictedOutcome(inputs);
			// System.out.print(Arrays.toString(inputs));
			// System.out.print("->");
			// System.out.println(Arrays.toString(predictedOutcome));
			adjustWeights(inputs);
			// if (wrapper.getIterations() % 1000 == 0) {
			// System.out.println(name + " did " + wrapper.getIterations() + "
			// iterations...");
			// }
		} while (wrapper.getIterations() < maxIterations);
		System.out.println(name + " on thread " + Thread.currentThread().getName() + " done.");
	}

	private void adjustWeights(double[] inputs) {
		double maxInput = Double.MIN_VALUE;
		for (int i = 0; i < inputs.length; i++) {
			if (inputs[i] > maxInput) {
				maxInput = inputs[i];
			}
		}
		double[] desiredOutcome;
		if (maxInput < 0.5) {
			desiredOutcome = new double[] { 1.0, 0.0 };
		} else {
			desiredOutcome = new double[] { 0.0, 1.0 };
		}
		// System.out.println("Train to " + Arrays.toString(desiredOutcome));
		wrapper.adjustWeights(inputs, desiredOutcome);
	}

	@Override
	public String toString() {
		return name + " did " + wrapper.getIterations() + " iterations...";
	}
}
