package svm.simplesmo;

import java.util.HashSet;
import java.util.Random;

import svm.model.Model;

public class SMO {
	private HashSet<Integer> boundAlpha = new HashSet<Integer>();
	private Random random = new Random();

	private double x[][];
	int y[];
	double a[];
	double b = 0.0;
	double kernel[][];

	/**
	 *
	 * @param x训练数据
	 * @param y训练数据类别
	 * @return 训练结果，分类函数;
	 */
	public Model train(double x[][], int y[]) {
		this.x = x;
		this.y = y;
		kernel = new double[x.length][x.length];
		initiateKernel(x.length);

		double C = 1;//对不在界内的惩罚因子
		double tol = 0.01;//容忍极限值
		int maxPasses = 5; //表示没有改变拉格朗日乘子的最多迭代次数

		double[] a = new double[x.length];///拉格朗日乘子
		this.a = a;
		//将乘子初始化为0
		for (int i = 0; i < x.length; i++) {
			a[i] = 0;
		}

		int passes = 0;
		while (passes < maxPasses) {
			int num_changed_alphas = 0;

			for (int i = 0; i < x.length; i++) {

				double Ei = getE(i);
				if ((y[i] * Ei < -tol && a[i] < C) || (y[i] * Ei > tol && a[i] > 0)) {
					int j;
					if (this.boundAlpha.size() > 0) {
						j = findMax(Ei, this.boundAlpha);
					} else {
						j = RandomSelect(i);
						double Ej = getE(j);
						double oldAi = a[i];
						double oldAj = a[j];
						double L, H;

						if (y[i] != y[j]) {
							L = Math.max(0, a[j] - a[i]);
							H = Math.min(C, C - a[i] + a[j]);
						} else {
							L = Math.max(0, a[i] + a[j] - C);
							H = Math.min(C, a[j] + a[i]);
						}

						double eta = 2 * k(i, j) - k(i, i) - k(j, j);
						if (eta >= 0)
							continue;
						a[j] = a[j] - y[j] * (Ei - Ej) / eta;
						//对a[j]调整之后判断是否有0<a[j]<C，若成立，则该点是支持向量，加入边界点集中
						if (0 < a[j] && a[j] < C)
							this.boundAlpha.add(j);

						if (a[j] < L)
							a[j] = L;
						else if (a[j] > H)
							a[j] = H;

						if (Math.abs(a[j] - oldAj) < 1e-5)
							continue;
						a[i] = a[i] + y[i] * y[j] * (oldAj - a[j]);
						if (0 < a[i] && a[i] < C)
							this.boundAlpha.add(i);

						double b1 = b - Ei - y[i] * (a[i] - oldAi) * k(i, j) - y[j] * (a[j] - oldAj) * k(i, j);
						double b2 = b - Ej - y[i] * (a[i] - oldAi) * k(i, j) - y[j] * (a[j] - oldAj) * k(i, j);

						if (0 < a[i] && a[i] < C)
							b = b1;
						else if (0 < a[j] && a[j] < C)
							b = b2;
						else
							b = (b1 + b2) / 2;

						num_changed_alphas = num_changed_alphas + 1;
					}
				}

				if (num_changed_alphas == 0) {
					passes++;
				} else
					passes = 0;
			}
		}

		return new Model(a, y, b);
	}

	private double getE(int i) {
		return f(i) - y[i];
	}

	private double f(int j) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += a[i] * y[i] * kernel[j][i];
		}
		return sum + this.b;
	}

	private void initiateKernel(int length) {
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++) {
				kernel[i][j] = k(i, j);
			}
		}
	}

	private int findMax(double Ei, HashSet<Integer> boundAlpha2) {
		double max = 0;
		int maxIndex = -1;
		for (Integer j : boundAlpha2) {
			double Ej = getE(j);
			if (Math.abs(Ei - Ej) > max) {
				max = Math.abs(Ei - Ej);
				maxIndex = j;
			}
		}
		return maxIndex;
	}

	private int RandomSelect(int i) {
		int j;
		do {
			j = random.nextInt(x.length);
		} while (i == j);
		return j;
	}

	private double k(int i, int j) {
		double sum = 0.0;
		for (int t = 0; t < x[i].length; t++) {
			sum += x[i][t] * x[j][t];
		}
		return sum;
	}

	public int[] predict(Model model, double x[][]) {
		int y[] = new int[x.length];
		for (int i = 0; i < x.length; i++) {
			int len = model.getY().length;
			double sum = 0;
			for (int j = 0; j < len; j++) {
				sum += model.getY()[j] * model.getA()[j] * k(j, i);
			}
			sum += model.getB();
			if (sum > 0)
				y[i] = 1;
			else if (sum < 0)
				y[i] = -1;
			else
				y[i] = 0;//y[i]=0,表示处于超平面上;
		}
		return y;
	}
}
