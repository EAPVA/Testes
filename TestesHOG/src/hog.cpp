#include "hog.h"
#include "hog_constants.h"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

#define INPUT_PATH "resources/images"

int main(int argc, char **argv) {

  std::vector<std::string> image_list = getImagesList(INPUT_PATH);

  cv::Mat img;
  cv::Mat out;

  for (int i = 0; i < image_list.size(); ++i) {
    std::cout << image_list[i] << std::endl;
    img = cv::imread(image_list[i], CV_LOAD_IMAGE_GRAYSCALE);
    out = draw_hog(calc_hog(img));
    cv::imwrite(image_list[i] + "_hist.png", out);
    break;
  }
}

hog calc_hog(cv::Mat img) {
  hog ret;

  cv::Mat x_mag, y_angle;

  cv::Sobel(img, x_mag, -1, 1, 0, 1);
  cv::Sobel(img, y_angle, -1, 0, 1, 1);

  cv::cartToPolar(x_mag, y_angle, x_mag, y_angle, true);

  float bin_size = 360.0 / HOG_NUMBER_OF_BINS;

  int n_cells = HOG_NUMBER_OF_BLOCKS * HOG_CELLS_PER_BLOCK;

  int top_row = 0;
  int bottom_row = (img.rows / HOG_GRID_HEIGHT) - 1;
  int extra_rows = (img.rows % HOG_GRID_HEIGHT);


  int left_col = 0;
  int right_col = (img.cols / HOG_GRID_WIDTH) - 1;
  int extra_cols = (img.cols % HOG_GRID_WIDTH);

  for (int i = 0; i < n_cells; ++i) {
    histogram hist;
    for (int j = 0; j < HOG_NUMBER_OF_BINS; ++j) {
      hist.bins.push_back(0.0f);
    }

    if (extra_rows) {
      bottom_row++;
      extra_rows--;
    }
    if (extra_cols) {
      right_col++;
      extra_cols--;
    }

    for (int j = top_row; j <= bottom_row; ++j) {
      for (int k = left_col; k <= right_col; ++k) {
        int left_bin = (int)floor(((y_angle.at(j,k) - (bin_size / 2)) / bin_size));
        if (left_bin < 0) left_bin += hist.bins.size();
        int right_bin = (left_bin + 1) % hist.bins.size();

        float delta = (y_angle.at(j,k) / bin_size) - right_bin;
        if (delta > 0) delta -= hist.bins.size();

        hist.bins[left_bin] += (0.5 - delta) * x_mag.at(j,k);
        hist.bins[right_bin] += (0.5 + delta) * x_mag.at(j,k);
      }
    }

    ret.cells.push_back(hist);
  }

  return ret;
}

cv::Mat draw_hog(hog h) {
  cv::Mat ret(HOG_DRAW_IMAGE_HEIGHT, HOG_DRAW_IMAGE_WIDTH, CV_8UC1);

  float max_abs = 0.0f;
  for (int i = 0; i < h.cells.size(); ++i) {
    for (int j = 0; j < h.cells[i].bins.size(); ++j) {
      if (max_abs < abs(h.cells[i].bins[j])) max_abs = abs(h.cells[i].bins[j]);
    }
  }

  int cell_height = ret.rows / h.cells.size();
  int bin_width = ret.cols / HOG_NUMBER_OF_BINS;
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
