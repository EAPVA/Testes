/*
 * histogram.h
 *
 *  Created on: Apr 1, 2015
 *      Author: teider
 */

#ifndef HOG_H_
#define HOG_H_

//#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

typedef struct histogram {
  std::vector<float> bins;
} histogram;

typedef struct hog {
  std::vector<histogram> cells;
  float label;
} hog;

hog calc_hog(cv::Mat img, float label);

cv::Mat draw_hog(hog h);

void generate_train_data(std::vector<hog> inputs, cv::Mat& train_data,
                         cv::Mat& labels);


#endif /* HOG_H_ */
