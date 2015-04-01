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
  //vector<string> bin_labels;
} hog;

hog calc_hog(cv::Mat img);

cv::Mat draw_hog(hog h);


#endif /* HOG_H_ */
