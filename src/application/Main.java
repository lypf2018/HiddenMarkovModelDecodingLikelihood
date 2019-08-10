/**
 * 
 */
package application;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import problem1b.HiddenMarkovModel;

/**
 * @author yzc
 *
 */
public class Main {

	enum StateEnum {
		START, HOT, COLD, END
	}

	enum ObservationEnum {
		ONE, TWO, THREE
	}

	/**
	 * @param args observationSequenceString. Only one argument allowed!
	 */
	public static void main(String[] args) {
		// Create Hidden Markov Model
		HiddenMarkovModel<StateEnum, ObservationEnum> hiddenMarkovModel = new HiddenMarkovModel<>();
		setupHiddenMarkovModel(hiddenMarkovModel);

		if (args.length > 1) {
			System.out.println("Only one argument allowed!");
		} else if (args.length == 1) {
			// Process input observation sequence
			String observationSequenceString = args[0];
			List<ObservationEnum> observationSequence = convertInputToObservationList(observationSequenceString);
			
			// Decoding
			List<StateEnum> mostLikelyStateSequence = hiddenMarkovModel.decoding(observationSequence);
			
			// Output result
			displayResult(observationSequenceString, mostLikelyStateSequence);
		} else {
			// Default homework 3 answer
			displayResult("331122313", hiddenMarkovModel.decoding(convertInputToObservationList("331122313")));
			displayResult("331123312", hiddenMarkovModel.decoding(convertInputToObservationList("331123312")));
		}
	}

	/**
	 * setup HiddenMarkovModel
	 * @param hiddenMarkovModel HiddenMarkovModel Object
	 */
	public static void setupHiddenMarkovModel(HiddenMarkovModel<StateEnum, ObservationEnum> hiddenMarkovModel) {
		// Set start and end state
		if (hiddenMarkovModel.getStartState() == null) {
			hiddenMarkovModel.setStartState(StateEnum.START);
		}
		if (hiddenMarkovModel.getEndState() == null) {
			hiddenMarkovModel.setEndState(StateEnum.END);
		}

		// Set start state data
		// Set start transition
		HashMap<StateEnum, Double> startTransition = new HashMap<>();
		startTransition.put(StateEnum.HOT, 0.8);
		startTransition.put(StateEnum.COLD, 0.2);
		hiddenMarkovModel.getTransitionProbabilityMap().put(StateEnum.START, startTransition);

		// Add HOT state data
		hiddenMarkovModel.getStateList().add(StateEnum.HOT);
		// Add HOT transition
		HashMap<StateEnum, Double> hotTransition = new HashMap<>();
		hotTransition.put(StateEnum.HOT, 0.7);
		hotTransition.put(StateEnum.COLD, 0.3);
		hotTransition.put(StateEnum.END, 1.0);
		hiddenMarkovModel.getTransitionProbabilityMap().put(StateEnum.HOT, hotTransition);
		// Add observation likelihood
		HashMap<ObservationEnum, Double> hotLikelihood = new HashMap<>();
		hotLikelihood.put(ObservationEnum.ONE, 0.2);
		hotLikelihood.put(ObservationEnum.TWO, 0.4);
		hotLikelihood.put(ObservationEnum.THREE, 0.4);
		hiddenMarkovModel.getObservationLikelihoodMap().put(StateEnum.HOT, hotLikelihood);

		// Add COLD state data
		hiddenMarkovModel.getStateList().add(StateEnum.COLD);
		// Add COLD transition
		HashMap<StateEnum, Double> coldTransition = new HashMap<>();
		coldTransition.put(StateEnum.HOT, 0.4);
		coldTransition.put(StateEnum.COLD, 0.6);
		coldTransition.put(StateEnum.END, 1.0);
		hiddenMarkovModel.getTransitionProbabilityMap().put(StateEnum.COLD, coldTransition);
		// Add observation likelihood
		HashMap<ObservationEnum, Double> coldLikelihood = new HashMap<>();
		coldLikelihood.put(ObservationEnum.ONE, 0.5);
		coldLikelihood.put(ObservationEnum.TWO, 0.4);
		coldLikelihood.put(ObservationEnum.THREE, 0.1);
		hiddenMarkovModel.getObservationLikelihoodMap().put(StateEnum.COLD, coldLikelihood);
	}

	/**
	 * Convert String to Integer list
	 * @param observationSequenceString Observation sequence string
	 * @return Integer list
	 */
	public static List<ObservationEnum> convertInputToObservationList(String observationSequenceString) {
		List<ObservationEnum> observationSequence = new ArrayList<>();
		for (int i = 0; i < observationSequenceString.length(); i++) {
			observationSequence.add(ObservationEnum.values()[Integer.parseInt(observationSequenceString.substring(i,i + 1)) - 1]);
		}
		return observationSequence;
	}

	/**
	 * Display decoded result
	 * @param observationSequenceString Observation sequence string
	 * @param mostLikelyStateSequence Result state sequence
	 */
	public static void displayResult(String observationSequenceString, List<StateEnum> mostLikelyStateSequence) {
		System.out.print(observationSequenceString);
		System.out.print(":");
		for (int i = 0; i < mostLikelyStateSequence.size(); i++) {
			System.out.print(mostLikelyStateSequence.get(i).name().charAt(0));
		}
		System.out.println();
	}
}
