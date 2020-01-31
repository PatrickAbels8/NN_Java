package test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import NeuralNetwork.InputNeuron;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.WorkingNeuron;
import test.MNISTDecoder.Digit;

public class MNISTLearn {
	public static List<Digit> digits;
	public static List<Digit> digitsTest;
	public static NeuralNetwork nn = new NeuralNetwork();
	public static InputNeuron[][] inputs = new InputNeuron[28][28]; //längen anpassen a b
	public static WorkingNeuron[] outputs = new WorkingNeuron[10]; // ...c
	
	public static void main(String[] args) throws IOException{
		digits = MNISTDecoder.loadDataSet("C:\\Users\\patri\\Desktop\\Java\\NeuralNetwork\\train-images.idx3-ubyte", "C:\\Users\\patri\\Desktop\\Java\\NeuralNetwork\\train-labels.idx1-ubyte");
		digitsTest = MNISTDecoder.loadDataSet("C:\\Users\\patri\\Desktop\\Java\\NeuralNetwork\\t10k-images.idx3-ubyte", "C:\\Users\\patri\\Desktop\\Java\\NeuralNetwork\\t10k-labels.idx1-ubyte");
		//längen inputs, outputs entsprechend der datasets anpassen 
		
		for(int i=0; i<28; i++) //längen entsprechend der inputs anpassen (a)
			for(int k=0; k<28; k++) // so (b)
				inputs[i][k] = nn.createNewInput();
		
		for(int i=0; i<10; i++) //längen entsprechend der outputs anpassen (c)
				outputs[i] = nn.createNewOutput();
		
		int numHiddenNeurons = 100;
		nn.createHiddenNeurons(numHiddenNeurons);
		
		//Random FullMash
		Random ran = new Random();
		float[] weights = new float[(28*28+10)*numHiddenNeurons]; //so (a*b*c)
		for(int i=0; i<weights.length; i++)
			weights[i] = ran.nextFloat();
		nn.createFullMesh(weights);
		
		//LearnDelta
		float epsilon = 0.0005f;
		while(true) {
			test();
			for(int i=0; i<digits.size(); i++) {
				for(int x=0; x<28; x++) //so
					for(int y=0; y<28; y++) // so
						inputs[x][y].setValue(MNISTDecoder.toUnsignedByte(digits.get(i).data[x][y])/255f);
				
				float[] shoulds = new float[10];
				shoulds[digits.get(i).label] = 1;
				nn.backpropagation(shoulds, epsilon);
			}
			epsilon *= 0.9f;
		}
	}
	
	public static void test() {
		int correct = 0;
		int incorrect = 0;
		
		for(int i=0; i<digitsTest.size(); i++) {
			nn.reset();
			for(int x=0; x<28; x++) //so
				for(int y=0; y<28; y++) // so
					inputs[x][y].setValue(MNISTDecoder.toUnsignedByte(digitsTest.get(i).data[x][y])/255f);
			
			ProbabilityDigit[] probs = new ProbabilityDigit[10];
			for(int k=0; k<probs.length; k++) {
				probs[k] = new ProbabilityDigit(k, outputs[k].getValue());
			}
			Arrays.sort(probs, Collections.reverseOrder());

			if(digitsTest.get(i).label == probs[0].DIGIT)
				correct++;
				else 
					incorrect++;
		}
		
		float percentage = (float)correct/(float)(correct+incorrect);
		System.out.println(percentage*100 +"% aller geratenen Ziffern stimmen.");
		System.out.println("Ich lerne wieder...");
	}
	
	public static class ProbabilityDigit implements Comparable<ProbabilityDigit>{
		public final int DIGIT;
		public float probability;
		
		public ProbabilityDigit(int digit, float probability) {
			this.DIGIT = digit;
			this.probability = probability;
		}

		@Override
		public int compareTo(ProbabilityDigit other) {
			return ((this.probability == other.probability)? 0: ((this.probability > other.probability)? 1: -1));
		}
	}
}
