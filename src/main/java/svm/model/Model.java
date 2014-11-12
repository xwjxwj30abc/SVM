package svm.model;

public class Model {

	private double a[];
	private int y[];
	private double b;

	public Model() {
		//
	}

	public Model(double a[], int y[], double b) {
		this.a = a;
		this.y = y;
		this.b = b;
	}

	public double[] getA() {
		return a;
	}

	public void setA(double[] a) {
		this.a = a;
	}

	public int[] getY() {
		return y;
	}

	public void setY(int[] y) {
		this.y = y;
	}

	public double getB() {
		return b;
	}

	public void setB(double b) {
		this.b = b;
	}

}
