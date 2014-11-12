package svm.simplesmo;

import svm.model.Model;
import svm.model.SVMData;
import svm.model.SVMFileReader;

public class SVM {

	public static void main(String[] args) {

		SVMFileReader reader = new SVMFileReader("heart_scale");
		SVMData svmData = reader.getSVMData(10);
		SMO smo = new SMO();
		Model model = smo.train(svmData.getX(), svmData.getY());
		System.out.println("OK");
		smo.predict(model, svmData.getX());
		for (int i = 0; i < 10; i++)
			System.out.println(smo.predict(model, svmData.getX())[i]);
		System.out.println("hello");
	}
}
