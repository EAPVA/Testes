#include "hog.h"
#include "hog_constants.h"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"

#define INPUT_PATH "resources/images"

int main(int argc, char **argv) {

  std::vector<std::string> image_list = getImagesList(INPUT_PATH);

  cv::Mat img;
  cv::Mat out;

  for (int i = 0; i < image_list.size(); ++i) {
    std::cout << image_list[i] << std::endl;
    out = draw_hog(calc_hog(img));
    cv::imwrite(image_list[i] + "_hist.png", out);
    break;
  }
}

hog calc_hog(cv::Mat img) {
  hog ret;

  int n_cells = NUMBER_OF_BLOCKS * CELLS_PER_BLOCK;

  for (int i = 0; i < n_cells; ++i) {
    histogram hist;
    for (int j = 0; j < NUMBER_OF_BINS; ++j) {
      hist.bins.push_back(i + j);
    }
    ret.cells.push_back(hist);
  }

  return ret;
}

cv::Mat draw_hog(hog h) {
  cv::Mat ret(HOG_DRAW_IMAGE_HEIGTH, HOG_DRAW_IMAGE_WIDTH, CV_8UC1);

  float max_abs = 0.0f;
  for (int i = 0; i < h.cells.size(); ++i) {
    for (int j = 0; j < h.cells[i].bins.size(); ++j) {
      if (max_abs < abs(h.cells[i].bins[j])) max_abs = abs(h.cells[i].bins[j]);
    }
  }

  int cell_height = ret.rows / h.cells.size();
  int bin_width = ret.cols / NUMBER_OF_BINS;
  float pixel_hvalue = (max_abs / (cell_height / 2));


  for (int i = 0; i < h.cells.size(); ++i) {
    int middle_row = cell_height / 2 + i * cell_height;
    for (int j = 0; j < h.cells[i].bins.size(); ++j) {
      cv::rectangle(ret, cv::Point_<int>(j * bin_width, middle_row),
                    cv::Point_<int>((j + 1) * bin_width - 1,
                                    middle_row - (h.cells[i].bins[j]) / pixel_hvalue), 255,
                                    CV_FILLED);
    }
  }

  return ret;
}
