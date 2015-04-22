#include "hog.h"
#include "hog_constants.h"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils.h"

#define INPUT_PATH "resources/images"

//#define DEBUG_L1
//#define DEBUG_L2
//#define DEBUG_PRINT_HIST

int main(int argc,
	char **argv)
{

	std::vector<std::string> image_list = getImagesList(INPUT_PATH);
	std::vector<float> labels = generateLabels(image_list);

	cv::Mat img;
	cv::Mat out;

	std::vector<hog> hog_list;

	int new_height = HOG_RESIZE_HEIGHT;

	std::cout << "Reading images: " << std::endl;
	for(int i = 0; i < image_list.size(); ++i)
	{
		std::cout << i << ": " << image_list[i] << "  " << labels[i]
			<< std::endl;
		img = cv::imread(image_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		img.convertTo(img, CV_32FC1);
		double scaleV = (double)new_height / (double)img.rows;
		double scaleH = scaleV * img.cols;
		cv::resize(img, img, cv::Size(), scaleH, scaleV, cv::INTER_LINEAR);
		hog_list.push_back(calc_hog(img, labels[i]));
#ifdef DEBUG_PRINT_HIST
		//out = draw_hog(hog_list.back());
		//cv::imwrite("outputs/" + image_list[i], out);
#endif //DEBUG_PRINT_HIST
	}
	cv::Mat train_data, label_data;
	generate_train_data(hog_list, train_data, label_data);

	CvSVMParams params;
	CvSVM svm;

	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

	std::cout << "Training... " << std::endl;

	svm.train_auto(train_data, label_data, cv::Mat(), cv::Mat(), params, 10);

	int true_positives = 0;
	int true_negatives = 0;
	int false_positives = 0;
	int false_negatives = 0;
	int total_positives = 0;
	int total_negatives = 0;

	std::cout << "Testing images... " << std::endl;
	std::cout << "Listing errors: " << std::endl;
	for(int i = 0; i < image_list.size(); ++i)
	{
		float result = svm.predict(train_data.row(i));
		float expected = label_data.at<float>(i);
		if(expected == 1.0)
		{
			++total_positives;
		} else
		{
			++total_negatives;
		}
		if(result == expected)
		{
			if(expected == 1.0)
			{
				++true_positives;
			} else
			{
				++true_negatives;
			}
		} else
		{
			if(expected == 1.0)
			{
				++false_negatives;
			} else
			{
				++false_positives;
			}
			std::cout << i << ": " << image_list[i] << "  ";
			std::cout << "expected: " << expected << "  ";
			std::cout << "obtained: " << result << std::endl;
		}
	}
	std::cout << std::endl;

	std::cout << "Total positives: " << total_positives << std::endl;
	std::cout << "Total negatives: " << total_negatives << std::endl;
	std::cout << "True positives: " << true_positives << std::endl;
	std::cout << "True negatives: " << true_negatives << std::endl;
	std::cout << "False positives: " << false_positives << std::endl;
	std::cout << "False negatives: " << false_negatives << std::endl;

	double precision = (true_positives)
		/ (double)(true_positives + false_positives);
	double recall = (true_positives)
		/ (double)(true_positives + false_negatives);

	double f_measure = 2 / ((1 / precision) + (1 / recall));

	std::cout << "Precision: " << precision << std::endl;
	std::cout << "Recall: " << recall << std::endl;
	std::cout << "f-Measure: " << f_measure << std::endl;
}

hog calc_hog(cv::Mat img,
	float label)
{
	hog ret;
	ret.label = label;

	cv::Mat img_dx(img.rows, img.cols, img.type());
	cv::Mat img_dy(img.rows, img.cols, img.type());

	cv::Mat img_mag(img.rows, img.cols, img.type());
	cv::Mat img_angle(img.rows, img.cols, img.type());

	//cv::Mat dx = (cv::Mat_<float>(3,3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
	//cv::Mat dy = (cv::Mat_<float>(3,3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);

	//cv::filter2D(img, img_dx, -1, dx, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
	//cv::filter2D(img, img_dy, -1, dy, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
	cv::Sobel(img, img_dx, -1, 1, 0, 1);
	cv::Sobel(img, img_dy, -1, 0, 1, 1);

	cv::cartToPolar(img_dx, img_dy, img_mag, img_angle, true);

	float bin_size = 360.0 / HOG_NUMBER_OF_BINS;

	int n_cells = HOG_NUMBER_OF_BLOCKS * HOG_CELLS_PER_BLOCK;

	int top_row = 0;
	int bottom_row = (img.rows / HOG_GRID_HEIGHT) - 1;
	int extra_rows = (img.rows % HOG_GRID_HEIGHT);

	int left_col = 0;
	int right_col = (img.cols / HOG_GRID_WIDTH) - 1;
	int extra_cols = (img.cols % HOG_GRID_WIDTH);

	for(int i = 0; i < n_cells; ++i)
	{
		histogram hist;
		for(int j = 0; j < HOG_NUMBER_OF_BINS; ++j)
		{
			hist.bins.push_back(0.0f);
		}

		if(extra_rows)
		{
			bottom_row++;
			extra_rows--;
		}
		if(extra_cols)
		{
			right_col++;
			extra_cols--;
		}

		float bin_total = 0.0f;

		for(int j = top_row; j <= bottom_row; ++j)
		{
			for(int k = left_col; k <= right_col; ++k)
			{
				if(img_mag.at<float>(j, k) > 0)
				{
					int left_bin =
						(int)floor(
							((img_angle.at<float>(j, k) - (bin_size / 2))
								/ bin_size));
					if(left_bin < 0)
						left_bin += hist.bins.size();
					int right_bin = (left_bin + 1) % hist.bins.size();

					float delta = (img_angle.at<float>(j, k) / bin_size)
						- right_bin;
					if(delta > 1.0)
						delta -= hist.bins.size();
					delta = 0;

					hist.bins[left_bin] += (0.5 - delta)
						* img_mag.at<float>(j, k);
					hist.bins[right_bin] += (0.5 + delta)
						* img_mag.at<float>(j, k);
					bin_total += img_mag.at<float>(j, k);

#ifdef DEBUG_L2
					std::cout << "Processing pixel (" << j << ", " << k << "):" << std::endl;
					std::cout << "Original value: " << img.at<float>(j,k) << std::endl;
					std::cout << "dx: " << img_dx.at<float>(j,k) << "   ";
					std::cout << "dy: " << img_dy.at<float>(j,k) << std::endl;
					std::cout << "Magnitude: " << img_mag.at<float>(j,k) << "   ";
					std::cout << "Angle: " << img_angle.at<float>(j,k) << std::endl;
					std::cout << "Left bin: " << left_bin << "     Right bin: " << right_bin << "   ";
					std::cout << "Delta: " << delta << std::endl;

					std::cout << "hist[" << left_bin << "]: " << hist.bins[left_bin] << "   ";
					std::cout << "hist[" << right_bin << "]: " << hist.bins[right_bin] << "   ";
					std::cout << std::endl;
					std::cout << std::endl;
#endif //DEBUG_L2
				}
			}
		}

#ifdef DEBUG_L2
		for (int l = 0; l < hist.bins.size(); ++l)
		{
			std::cout << "hist[" << l << "]: " << hist.bins[l] << "   ";
		}
		std::cout << std::endl;
		std::cout << std::endl;
#endif //DEBUG_L2
		for(int i = 0; i < hist.bins.size(); ++i)
		{
			hist.bins[i] /= bin_total;
		}

#ifdef DEBUG_L1
		for (int i = 0; i < hist.bins.size(); ++i)
		{
			std::cout << "hist[" << i << "]: " << hist.bins[i] << "   ";
		}
		std::cout << std::endl;
		std::cout << std::endl;
#endif //DEBUG_L1
		ret.cells.push_back(hist);
	}

	return ret;
}

cv::Mat draw_hog(hog h)
{
	cv::Mat ret(HOG_DRAW_IMAGE_HEIGHT, HOG_DRAW_IMAGE_WIDTH, CV_8U);
	cv::rectangle(ret, cv::Rect_<int>(0, 0, ret.cols, ret.rows), 0, CV_FILLED);

	int cell_height = ret.rows / h.cells.size();
	int bin_width = ret.cols / HOG_NUMBER_OF_BINS;
	int color = 255;

	for(int i = 0; i < h.cells.size(); ++i)
	{
		float max_bin = 0.0f;
		for(int j = 0; j < h.cells[i].bins.size(); ++j)
		{
			if(max_bin < h.cells[i].bins[j])
				max_bin = h.cells[i].bins[j];
		}
		for(int j = 0; j < h.cells[i].bins.size(); ++j)
		{
			if(h.cells[i].bins[j] > 0)
			{
				int bin_height = (h.cells[i].bins[j] / max_bin) * cell_height;
				cv::rectangle(ret,
					cv::Rect_<int>(j * bin_width,
						(i + 1) * cell_height - bin_height, bin_width,
						bin_height), color, CV_FILLED);
			}
			if(color == 255)
				color = 100;
			else
				color = 255;
		}
	}
	return ret;
}

void generate_train_data(std::vector<hog> inputs,
	cv::Mat& train_data,
	cv::Mat& labels)
{
	int total_bins = HOG_NUMBER_OF_BLOCKS * HOG_CELLS_PER_BLOCK
		* HOG_NUMBER_OF_BINS;
	train_data.create(inputs.size(), total_bins, CV_32FC1);
	labels.create(1, inputs.size(), CV_32FC1);
	float* label_handle = labels.ptr<float>(0);
	for(int i = 0; i < inputs.size(); ++i)
	{
		float* tdata_handle = train_data.ptr<float>(i);
		for(int j = 0; j < inputs[i].cells.size(); ++j)
		{
			for(int k = 0; k < inputs[i].cells[j].bins.size(); ++k)
			{
				(*tdata_handle) = inputs[i].cells[j].bins[k];
				++tdata_handle;
			}
		}
		label_handle[i] = inputs[i].label;
	}

#ifdef DEBUG_L1
	for (int i = 0; i < train_data.rows; ++i)
	{
		float* ret_handle = train_data.ptr<float>(i);
		int pos = 0;
		for (int j = 0; j < train_data.cols; ++j)
		{
			std::cout << "train_data[" << i << "][" << pos << "]: " << ret_handle[pos] << "   ";
			++pos;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
#endif //DEBUG_L1
}
