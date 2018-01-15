package net.b0n541.deeplearning4j.parallel;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DL4JParallelTest {

	public static void main(String[] args) throws InterruptedException {

		List<NetTrainer> trainer = IntStream.range(0, 10)
				.mapToObj(i -> new NetTrainer("Trainer " + i, 100000L))
				.collect(Collectors.toList());

		ExecutorService executor = Executors.newFixedThreadPool(trainer.size());

		List<CompletableFuture<Void>> futures = trainer.stream()
				.map(t -> CompletableFuture.runAsync(() -> t.trainNet()))
				.collect(Collectors.toList());

		// List<Void> result = futures.stream()
		// .map(CompletableFuture::join)
		// .collect(Collectors.toList());
		do {
			trainer.stream().forEach(System.out::println);
			Thread.sleep(1000);
		} while (true);
	}
}
