package test;

import NeuralNetwork.InputNeuron;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.WorkingNeuron;

public class SingleLayerPerceptron {
	
	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork();
		
		//inputs 
		InputNeuron in1 = nn.createNewInput();
		InputNeuron in2 = nn.createNewInput();
		InputNeuron in3 = nn.createNewInput();
		InputNeuron in4 = nn.createNewInput();
		
		//hiddens
		nn.createHiddenNeurons(3);
		
		//outputs
		WorkingNeuron out1 = nn.createNewOutput();
		
		nn.createFullMesh(30, -1, 2, 0, 30, -1, 2, 0, 30, -1, 2, 0, 30, -1, 2);
		
		
		in1.setValue(1);
		in2.setValue(2);
		in3.setValue(3);
		in4.setValue(4);
		
		System.out.println(out1.getValue());
	}
}
